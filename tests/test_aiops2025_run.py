"""scripts/aiops2025_run.py on the synthetic release: --workers n gives the same answers as one process (the partition does not change
any result), resume keeps finished cases and re-runs an answer that lost its row, an errored case writes no answer, and the merge
and the scorer see one answer per case. No real data.

    python tests/test_aiops2025_run.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ["DATASET_PROFILE"] = "aiops2025"

import pandas as pd   # noqa: E402

import test_aiops2025 as syn   # noqa: E402

RUN = [sys.executable, os.path.join(ROOT, "scripts", "aiops2025_run.py")]
T = []


def test(fn):
    T.append(fn)
    return fn


def run(args, check=True):
    p = subprocess.run(RUN + args, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    if check:
        assert p.returncode == 0, p.stdout[-800:] + p.stderr[-800:]
    return p


def answers(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            out[d["uuid"]] = (d["component"], d["reason"], len(d["reasoning_trace"]))
    return out


def main_tests(rel, tmp):
    @test
    def workers_give_the_same_answers_as_one_process():
        one, many = os.path.join(tmp, "one"), os.path.join(tmp, "many")
        run(["--data", rel, "--out", one])
        run(["--data", rel, "--out", many, "--workers", "2"])
        a, b = answers(os.path.join(one, "AIOPS_GAMA_core.jsonl")), answers(os.path.join(many, "AIOPS_GAMA_core.jsonl"))
        assert len(a) == 2 and a == b, (a, b)
        csv = pd.read_csv(os.path.join(many, "AIOPS_GAMA_core.csv"))
        assert len(csv) == 2 and "peak_rss_mb" in csv.columns and csv["peak_rss_mb"].gt(0).all()

    @test
    def every_run_leaves_a_zip_and_four_sessions_merge_from_their_zips():
        import zipfile
        many = os.path.join(tmp, "many")
        z = zipfile.ZipFile(os.path.join(many, "AIOPS_GAMA_core_all.zip"))
        names = z.namelist()
        assert "AIOPS_GAMA_core.jsonl" in names and any(n.endswith("shard0of2.jsonl") for n in names), names
        for k in (0, 1):                                                         # two sessions, one shard each
            run(["--data", rel, "--out", os.path.join(tmp, f"sess{k}"), "--shard", f"{k}/2"])
            assert os.path.exists(os.path.join(tmp, f"sess{k}", f"AIOPS_GAMA_core_shard{k}of2_all.zip"))
        merged = os.path.join(tmp, "merged")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "aiops2025_merge.py"), "--inputs",
                            os.path.join(tmp, "sess0", "*_all.zip"), os.path.join(tmp, "sess1", "*_all.zip"), "--out", merged,
                            "--name", "AIOPS_GAMA_core"], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-500:]
        assert answers(os.path.join(merged, "AIOPS_GAMA_core.jsonl")) == answers(os.path.join(tmp, "one", "AIOPS_GAMA_core.jsonl"))

    @test
    def resume_keeps_finished_cases_and_reruns_an_answer_without_its_row():
        out = os.path.join(tmp, "resume")
        run(["--data", rel, "--out", out])
        jl, csvp = os.path.join(out, "AIOPS_GAMA_core.jsonl"), os.path.join(out, "AIOPS_GAMA_core.csv")
        before = answers(jl)
        lines = open(jl, encoding="utf-8").read().splitlines()
        open(jl, "w", encoding="utf-8").write(lines[0] + "\n")                       # one answer lost
        p = run(["--data", rel, "--out", out])
        assert "resume: 1 cases kept" in p.stdout, p.stdout[-400:]
        assert answers(jl) == before and len(pd.read_csv(csvp)) == 2
        p2 = run(["--data", rel, "--out", out])
        assert "resume: 2 cases kept" in p2.stdout and answers(jl) == before        # nothing left to do, nothing duplicated
        assert len([ln for ln in open(jl, encoding="utf-8") if ln.strip()]) == 2

    @test
    def an_errored_case_writes_no_answer_and_is_retried():
        import scripts.aiops2025_run as R
        bad = {"uuid": "zzzz", "description": "", "start": None, "end": None}
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):                              # the expected traceback
            uuid, row, ans, chks, cov, err, _ = R.worker_case((bad, rel, [], False, "gama", "core", False, None))
        assert ans is None and err and row["error"] and uuid == "zzzz"
        out = os.path.join(tmp, "err")
        run(["--data", rel, "--out", out])
        df = pd.read_csv(os.path.join(out, "AIOPS_GAMA_core.csv"))
        pd.concat([df, pd.DataFrame([{"case_id": "aaaa0001-1", "error": "boom"}])]).iloc[[0, 2]].to_csv(
            os.path.join(out, "AIOPS_GAMA_core.csv"), index=False)                    # first case: row replaced by an errored one
        p = run(["--data", rel, "--out", out])
        assert answers(os.path.join(out, "AIOPS_GAMA_core.jsonl")).keys() == {"aaaa0001-1", "bbbb0002-2"}, p.stdout[-400:]

    @test
    def merge_and_scorer_take_one_answer_per_case():
        out = os.path.join(tmp, "many")
        from scripts import aiops2025_score as S
        preds = [json.loads(ln) for ln in open(os.path.join(out, "AIOPS_GAMA_core.jsonl"), encoding="utf-8")]
        preds.insert(0, {"uuid": preds[0]["uuid"], "component": "", "reason": "", "reasoning_trace": []})   # old empty line first
        gts = [syn_gt(p["uuid"], p["component"]) for p in preds[1:]]
        d, s = S.score(preds, gts)
        assert s["n"] == 2 and s["LA"] == 1.0, s


def syn_gt(uuid, comp):
    return {"uuid": uuid, "fault_category": "stress test", "fault_type": "cpu stress", "instance_type": "service", "service": comp,
            "instance": comp, "key_observations": [], "key_metrics": ["pod_cpu_usage"], "fault_description": ["cpu stress"]}


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="aiops_run_test_")
    try:
        rel = os.path.join(tmp, "release")
        syn.write_release(rel)
        main_tests(rel, tmp)
        ok = 0
        for fn in T:
            try:
                fn()
                print("ok  ", fn.__name__)
                ok += 1
            except Exception as e:      # noqa: BLE001
                print("FAIL", fn.__name__, repr(e)[:600])
        print(f"{ok}/{len(T)} passed")
        sys.exit(0 if ok == len(T) else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
