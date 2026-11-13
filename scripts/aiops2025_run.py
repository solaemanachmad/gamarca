#!/usr/bin/env python3
"""GAMA-RCA on AIOps2025, zero-shot (PREREGISTRATION_AIOPS2025.md). Reads input.json and the telemetry; never opens ground truth.

    python scripts/aiops2025_run.py --data /kaggle/input/aiops2025 --out /kaggle/working --shard 0/4            # diagnose
    python scripts/aiops2025_run.py --data ... --check --cases first:5                                         # label-free checks

Writes <name>.jsonl (one answer per case, the benchmark's submission format: uuid, component, reason, time, reasoning_trace),
<name>.csv (decision statistics per case), <name>.manifest.json. --check writes <name>_check.csv: adapter coverage and, per entity
kind and signal, how many (entity, signal) pairs are testable in the window and how many hold (false-alarm view).

LLM arms (API backbone on a CPU session, the key from the Kaggle Secret named in the preset, never printed):
    python scripts/aiops2025_run.py --modules hybrid --model gptoss120b_cerebras --shard 0/4         # GAMA Hybrid
    python scripts/aiops2025_run.py --arm evidence_llm --model gptoss120b_cerebras --shard 0/4       # Evidence->LLM reference
--model takes a key of MODELS in scripts/kaggle_run.py (the same presets as the RCA100 runs). Re-running the same command resumes:
cases already answered without error are kept.
"""
import os
import subprocess
import sys

os.environ.setdefault("DATASET_PROFILE", "aiops2025")       # must precede every import of the core
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":   # deterministic set / dict order (as kaggle_job -> main.py)
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def _model_env(argv):
    """--model KEY: the env of scripts/kaggle_run.py BASE + MODELS[KEY] + FIXED (as scripts/kaggle_job.py), set BEFORE the core reads
    config; the API key from the environment or the Kaggle Secret named in the preset (never printed, never written)."""
    if "--model" not in argv:
        return None, {}
    key = argv[argv.index("--model") + 1]
    sys.path.insert(0, HERE)
    from kaggle_job import definitions
    d = definitions("")
    assert key in d["MODELS"], f"unknown model {key!r}; keys: {', '.join(d['MODELS'])}"
    env = {**{k: v for k, v in d["BASE"].items() if k != "RCA100_ROOT"}, **d["MODELS"][key], **d["FIXED"]}
    os.environ.update({k: str(v) for k, v in env.items()})
    secret = d["MODELS"][key].get("API_KEY_SECRET")
    if secret:
        api_key = os.environ.get(secret)
        if not api_key:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret(secret)
        assert api_key, f"Kaggle Secret {secret} is empty"
        os.environ["API_KEY"] = api_key
    return key, d["MODELS"][key]


MODEL, MODEL_ENV = _model_env(sys.argv) if __name__ == "__main__" else (None, {})
for _mod, _pip in (("networkx", "networkx"), ("drain3", "drain3"), ("pyarrow", "pyarrow")):
    try:
        __import__(_mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", _pip])

import argparse          # noqa: E402
import json              # noqa: E402
import platform          # noqa: E402
import time              # noqa: E402
import traceback         # noqa: E402
from collections import Counter   # noqa: E402

import pandas as pd      # noqa: E402

from data.dataset_profile import PROFILE   # noqa: E402
from data.aiops2025_profile import phrase  # noqa: E402
from adapters.aiops2025 import Release   # noqa: E402
from gama.config import Modules            # noqa: E402
from gama.core.belief import modality           # noqa: E402

ACTION = {"metrics": "LoadMetrics", "traces": "TraceAnalysis", "logs": "LogSearch", "events": "LoadEvents"}
MAX_STEPS = 5                               # A6: short reasoning trace (Efficiency)


def _llm_evidence(tr, comp) -> list:
    """Evidence->LLM / Belief->LLM arms: the holding observations the prompt showed, of the answered component first (same format as
    the GAMA arms, so the trace is scored alike)."""
    by = tr.get("evidence_by_entity") or {}
    out = []
    for ent in [comp] + [e for e in by if e != comp]:
        for line in by.get(ent, []):
            sig, _, val = str(line).partition(": ")
            out.append({"entity": ent, "signal": sig, "ev": val, "holds": True})
    return out


def answer(case, r) -> dict:
    tr = r.retrieval_stats.get("_trace") or {}
    comp = r.predicted_entity_ids[0] if r.predicted_entity_ids else ""
    if "evidence_by_bayes_factor" in tr:
        ev = [e for e in (tr.get("evidence_by_bayes_factor") or []) if e.get("holds")][:MAX_STEPS]
    else:
        ev = _llm_evidence(tr, comp)[:MAX_STEPS]
    reason = phrase(r.predicted_fault_type) + (f" at {comp}" if comp else "")
    if ev:
        reason += "; " + "; ".join(f"{e['entity']} {e['signal']}" for e in ev[:3])
    steps = [{"step": i + 1, "action": f"{ACTION[modality(e['signal'])]}({e['entity']})", "observation": str(e["ev"])[:160]}
             for i, e in enumerate(ev)]
    return {"uuid": case.case_id, "component": comp, "reason": reason, "time": case.alert.raw["alert_window"]["start"],
            "reasoning_trace": steps}


def check(case, m) -> list:
    """Label-free: testable / holding continuous signals per (entity kind, signal) in the window."""
    from gama.core import evidence
    from gama.core.chain import callee_map
    from pipeline.calibration.fault import calibrate
    import config
    config.ALERT_T0_MODE = "window"                     # as in gama/run.run_case: evidence window = the case window
    db = evidence.signal_db(case, case.alert.alert_timestamp, m)
    if "services" not in db:
        return [{"case": case.case_id, "error": db.get("error")}]
    calibrate(db, peer=True, robust_null=False, ops_level=False, selfnorm=True, ratio=False, callees=callee_map(case.topology))
    kind = {e: ("node" if e in db["nodes"] else ("pod" if e in db.get("memberships", {}) else "service"))
            for e in list(db["services"]) + list(db["nodes"])}
    tot, hold, ent = Counter(), Counter(), Counter(kind.values())
    for e, sigs in db["_cal"].items():
        for s, r in sigs.items():
            if r.get("p") is not None:
                tot[(kind.get(e, "?"), s)] += 1
                hold[(kind.get(e, "?"), s)] += r["p"] <= 0.05
    return [{"case": case.case_id, "kind": k, "signal": s, "entities": ent[k], "testable": n, "holds": hold[(k, s)]}
            for (k, s), n in sorted(tot.items())]


_REL = None                                            # the release index, built once per process (os.walk of the whole release is slow)


def _peak_mb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024 if sys.platform == "darwin" else 1024), 0)


def worker_case(task_item):
    """Executes a single AIOps case for sequential or parallel worker pools."""
    spec, data_dir, all_windows, is_check, arm_name, mod_spec, need_llm, model_key = task_item
    t0 = time.time()
    try:
        from adapters.aiops2025 import Release as Rel, AIOpsCase as Case
        rel_obj = _REL or Rel(data_dir)
        case = Case(rel_obj, spec, all_windows=all_windows)
        cov = case.coverage()
        from gama.config import Modules as Mods
        m_obj = Mods.parse(mod_spec)
        if is_check:
            chks = check(case, m_obj)
            row = {"case_id": spec["uuid"], **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in cov.items()},
                   "wall_s": round(time.time() - t0, 1)}
            return spec["uuid"], row, None, chks, cov, None, round(time.time() - t0, 1)

        llm_client = None
        if need_llm:
            from agents.factory import get_llm_client
            llm_client = get_llm_client()

        if arm_name == "gama":
            from gama.run import run_case
            r = run_case(case, llm_client, m_obj)
        else:
            from baselines.belief_llm import run_belief_llm
            r = run_belief_llm(case, llm_client, with_belief=arm_name == "belief_llm")

        ans = answer(case, r)
        clean_row = {
            "case_id": spec["uuid"],
            "component": ans["component"],
            "fault_type": r.predicted_fault_type,
            "confidence": round(r.confidence, 4),
            "reason": ans["reason"],
            "n_steps": len(ans["reasoning_trace"]),
            "n_metrics": cov.get("n_metrics", 0),
            "n_logs": cov.get("n_logs", 0),
            "n_traces": cov.get("n_traces", 0),
            "load_time_s": cov.get("load_total_s", 0.0),
            "pipeline_time_s": r.retrieval_stats.get("total_pipeline_time_s", 0.0),
            "wall_s": round(time.time() - t0, 1),
            "peak_rss_mb": _peak_mb(),
        }
        return spec["uuid"], clean_row, ans, [], cov, None, round(time.time() - t0, 1)
    except Exception as exc:
        traceback.print_exc()
        err_msg = repr(exc)[:300]
        # an errored case writes NO answer line: it is retried on resume, and the scorer refuses a run with a missing case
        err_row = {"case_id": spec["uuid"], "error": err_msg, "wall_s": round(time.time() - t0, 1), "peak_rss_mb": _peak_mb()}
        return spec["uuid"], err_row, None, [], {}, err_msg, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/aiops2025")
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/kaggle/working"))
    ap.add_argument("--cases", default="all", help="all | first:N | uuid,uuid,...")
    ap.add_argument("--shard", default=None, help="k/n: the k-th of n contiguous blocks of the cases sorted by window start "
                                                  "(neighbouring cases share hourly / daily files, so the cache is reused)")
    ap.add_argument("--workers", type=int, default=1, help="n > 1: n independent shard processes (CPU arms only), each resumable and "
                                                          "restarted up to --retries times if it dies (e.g. out of memory); outputs merged")
    ap.add_argument("--retries", type=int, default=3, help="restarts per shard process when --workers > 1")
    ap.add_argument("--modules", default="core", help="GAMA arm: core | full | hybrid | core,-trace ...")
    ap.add_argument("--arm", default="gama", choices=("gama", "evidence_llm", "belief_llm"))
    ap.add_argument("--model", default=None, help="key of scripts/kaggle_run.py MODELS (LLM arms, modules with llm)")
    ap.add_argument("--check", action="store_true", help="label-free adapter / signal checks only (no diagnosis)")
    ap.add_argument("--dump-diagnostic", action="store_true", help="save separate <name>_diagnostic.csv with internal stats")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    assert PROFILE == "aiops2025", "DATASET_PROFILE must be aiops2025 before the core is imported"
    os.makedirs(a.out, exist_ok=True)
    rel = Release(a.data)
    specs = rel.cases()
    windows = [(s["start"], s["end"]) for s in specs]
    if a.cases.startswith("first:"):
        specs = specs[:int(a.cases.split(":")[1])]
    elif a.cases != "all":
        want = set(a.cases.split(","))
        specs = [s for s in specs if s["uuid"] in want]
    specs = sorted(specs, key=lambda s: (s["start"], s["uuid"]))     # order and partition do not change any result (cases are independent)
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        assert 0 <= k < n, "--shard k/n needs 0 <= k < n"
        size = -(-len(specs) // n)
        specs = specs[k * size:(k + 1) * size]
    m = Modules.parse(a.modules)
    needs_llm = a.arm != "gama" or m.llm
    assert a.check or not needs_llm or MODEL, "this arm uses an LLM: give --model (a key of scripts/kaggle_run.py MODELS)"
    assert a.workers == 1 or not (a.check or needs_llm or a.shard), \
        "--workers > 1 is for the CPU arm over all cases (LLM arms: one process, the API has a rate limit; use --shard across sessions)"
    if needs_llm and not a.check:
        from agents.factory import get_llm_client
        get_llm_client()
    commit = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True, text=True,
                            cwd=os.path.dirname(HERE)).stdout.strip()
    tag = a.modules if a.arm == "gama" else ""
    tag = tag.replace(",", "_").replace("-", "no")
    name = a.name or ("AIOPS_CHECK" if a.check else f"AIOPS_{a.arm.upper()}" + (f"_{tag}" if tag else "")
                      + (f"_{MODEL}" if needs_llm else "")) + (f"_shard{a.shard.replace('/', 'of')}" if a.shard else "")
    print(f"commit {commit} | profile {PROFILE} | arm {a.arm} | modules {m.spec()} | model {MODEL if needs_llm else '-'} | "
          f"{len(specs)} cases | output {name}", flush=True)
    manifest = {"commit": commit, "profile": PROFILE, "arm": a.arm, "modules": m.spec(), "model": MODEL if needs_llm else None,
                "model_env": MODEL_ENV if needs_llm else None, "cases": [s["uuid"] for s in specs], "shard": a.shard,
                "check": a.check, "python": platform.python_version(), "cpu_count": os.cpu_count(),
                "history_mask": os.environ.get("AIOPS_HISTORY_MASK", "none"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if a.workers > 1:
        return supervise(a, name)
    global _REL
    _REL = rel
    rows, checks, t_all = [], [], time.time()
    jl = None
    p_csv = os.path.join(a.out, f"{name}.csv")
    p_jl = os.path.join(a.out, f"{name}.jsonl")
    if not a.check:                                       # resume: a case is done when its csv row has no error and its answer exists
        done = set()
        if os.path.exists(p_csv):
            try:
                old = pd.read_csv(p_csv)
                ok = old[old["error"].isna()] if "error" in old.columns else old
                rows = ok.drop_duplicates("case_id", keep="last").to_dict("records")
                done = {str(r["case_id"]) for r in rows}
            except Exception:
                rows, done = [], set()
        kept = {}
        if os.path.exists(p_jl):
            for ln in open(p_jl, encoding="utf-8"):
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if str(rec.get("uuid")) in done:
                    kept[str(rec["uuid"])] = rec                      # last record of a case wins
        done &= set(kept)                                             # an answer without its row, or a row without answer: run again
        rows = [r for r in rows if str(r["case_id"]) in done]
        with open(p_jl, "w", encoding="utf-8") as f:
            for u in sorted(done):
                f.write(json.dumps(kept[u], ensure_ascii=False) + "\n")
        if done:
            print(f"resume: {len(done)} cases kept", flush=True)
        specs = [x for x in specs if x["uuid"] not in done]
        jl = open(p_jl, "a", encoding="utf-8")
    task_items = [(x, a.data, windows, a.check, a.arm, m.spec(), needs_llm, MODEL) for x in specs]
    for i, item in enumerate(task_items):
        uuid, row, ans, chks, cov, err, wall_s = worker_case(item)
        rows.append(row)
        if chks:
            checks.extend(chks)
        if jl and ans:
            jl.write(json.dumps(ans, ensure_ascii=False) + "\n")
            jl.flush()
        if not a.check:
            pd.DataFrame(rows).to_csv(p_csv, index=False)             # every case: a killed process loses nothing
        if err:
            print(f"[{i + 1}/{len(specs)}] {uuid} [error] {err}", flush=True)
        elif a.check:
            print(f"[{i + 1}/{len(specs)}] {uuid} {cov}", flush=True)
        else:
            print(f"[{i + 1}/{len(specs)} {(time.time() - t_all) / 60:5.1f} min] {uuid} -> {row.get('component')} "
                  f"{row.get('fault_type')} p={row.get('confidence', 0):.2f} ({wall_s:.0f}s, peak {row.get('peak_rss_mb', 0):.0f} MB)",
                  flush=True)
    if jl:
        jl.close()
    pd.DataFrame(rows).to_csv(p_csv, index=False)
    if checks:
        c = pd.DataFrame(checks)
        c.to_csv(os.path.join(a.out, f"{name}_check.csv"), index=False)
        if "signal" in c.columns:
            g = c.groupby(["kind", "signal"])[["entities", "testable", "holds"]].sum()
            g["testable_share"] = (g["testable"] / g["entities"]).round(3)
            g["hold_rate"] = (g["holds"] / g["testable"].clip(lower=1)).round(3)
            print(g.to_string())
    manifest.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=round(time.time() - t_all, 1),
                    n_errors=sum(1 for r in rows if r.get("error")))
    json.dump(manifest, open(os.path.join(a.out, f"{name}.manifest.json"), "w"), indent=1)
    subprocess.run(f"cd {a.out} && rm -f {name}_all.zip && zip -q {name}_all.zip {name}.* {name}_check.csv 2>/dev/null", shell=True)
    print("download:", os.path.join(a.out, f"{name}_all.zip"))
    n_err = sum(1 for r in rows if isinstance(r.get("error"), str))
    if n_err:
        print(f"{n_err} case(s) errored and have no answer; run the same command again to retry them", flush=True)
    return 1 if n_err else 0


def supervise(a, name) -> int:
    """--workers n: n independent shard processes of this script (own outputs, own resume); a shard that dies (out of memory, killed
    session) is restarted up to --retries times and continues where it stopped; afterwards the shards are merged. A killed worker
    cannot hang the others (it did with multiprocessing.Pool)."""
    drop = {"--workers", "--retries"}
    argv, skip = [], False
    for x in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if x in drop:
            skip = True
            continue
        if any(x.startswith(d + "=") for d in drop):
            continue
        argv.append(x)
    n = a.workers
    shard_dir = os.path.join(a.out, f"{name}_shards")
    os.makedirs(shard_dir, exist_ok=True)
    cmds = [[sys.executable, os.path.abspath(__file__), *argv, "--shard", f"{k}/{n}", "--out", shard_dir, "--name", f"{name}_shard{k}of{n}"]
            for k in range(n)]
    tries = [0] * n
    procs = {k: subprocess.Popen(cmds[k]) for k in range(n)}
    status = {}
    while procs:
        time.sleep(5)
        for k in list(procs):
            rc = procs[k].poll()
            if rc is None:
                continue
            del procs[k]
            if rc == 0:
                status[k] = "ok"
            elif tries[k] < a.retries:                               # died, or finished with errored cases: resume picks up
                tries[k] += 1
                print(f"[supervisor] shard {k}/{n} exited with {rc}; restart {tries[k]}/{a.retries}", flush=True)
                procs[k] = subprocess.Popen(cmds[k])
            else:
                status[k] = f"exit {rc}"
    print("[supervisor]", status, flush=True)
    merge = [sys.executable, os.path.join(HERE, "aiops2025_merge.py"), "--inputs", shard_dir, "--out", a.out, "--name", name]
    subprocess.run(merge)
    bad = {k: v for k, v in status.items() if v != "ok"}
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
