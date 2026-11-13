#!/usr/bin/env python3
"""aiops2025_score.py -- the AIOps2025 protocol of Cai et al. (2026), eq. (1)-(4), re-implemented (PREREGISTRATION_AIOPS2025.md A6).
The ONLY code that opens groundtruth.jsonl. No official scorer is part of the release; every reading of the published text that
needed a choice is listed under "Readings" below and printed with the result.

    python scripts/aiops2025_score.py --pred results/aiops2025/core/AIOPS_GAMA_core.jsonl --gt /path/to/groundtruth.jsonl
    python scripts/aiops2025_score.py --pred core=.../core.jsonl --pred hybrid=.../hybrid.jsonl --gt ... --out results/aiops2025/scores

Protocol (Cai et al. Section 4.4):
  LA   share of cases whose predicted component string-matches the ground truth exactly; pod-level faults need the exact pod;
       network attacks accept the source or the destination.                                                              eq. (1)
  TA   the reason field, truncated to its first 20 words, contains one of the ground-truth fault descriptions (keyword part only;
       the embedding fallback is unspecified, so TA is a lower bound).
  Exp  evidence points hit / evidence points expected; points = the union of key_metrics and key_observations; a point is hit when
       it appears in the first 20 characters of any observation of the reasoning trace.                                    eq. (2)
  Eff  min(1, exp(-(APL - 5) / 5)), APL = mean reasoning-trace length over the LA-correct cases.                           eq. (3)
  Final = (0.4 LA + 0.4 TA + 0.1 Exp + 0.1 Eff) x 100.                                                                     eq. (4)

Readings (fixed before any scored run):
  R1 Ground-truth fields as in the published example (Cai et al. Fig. 4): uuid, fault_category, fault_type, instance_type, service,
     instance, key_observations [{type, keyword: [...]}], key_metrics [...], fault_description [...]. `instance` may be a string or a
     list. For network attacks the accepted components are `instance` plus the fields named source / destination (and the
     spellings in NETWORK_FIELDS) when present. The field names found per fault category are printed (names only) so that the
     reading can be checked; unknown localisation fields are reported, never guessed.
  R2 Matching is exact after stripping surrounding whitespace (case-sensitive), as "string-matches exactly".
  R3 TA: lower-case, '-' and '_' read as spaces, whitespace collapsed; a description counts when it is a substring of the truncated
     reason.
  R4 Exp per evidence point: a key observation is one point, hit when any of its keywords is found; a key metric not already listed
     in a metric-type key observation is one more point (union). Metric names match as whole tokens ("rrt" does not hit "rrt_max");
     log keywords and trace nodes match as case-insensitive substrings. Exp is the mean of the per-case ratios (cases without
     evidence points are left out of the mean); the pooled ratio is reported next to it.
  R5 Eff is a corpus-level quantity. No LA-correct case -> Eff = 0.
  R6 Per-case Final (for paired tests and bootstrap) = 0.4 la + 0.4 ta + 0.1 exp_case + 0.1 Eff of that arm; its mean equals eq. (4)
     except for cases without evidence points. The 95% interval is a case bootstrap (5,000 resamples, seed 0) of eq. (4).
  R7 A run is scored only when it answers every ground-truth case (an errored case is an empty answer and scores 0). --partial
     overrides this for debugging; a partial score is not a result and using it is a deviation to disclose.
The per-case table contains ground truth: keep it private (it is git-ignored).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

W_LA, W_TA, W_EXP, W_EFF = 0.4, 0.4, 0.1, 0.1
REASON_WORDS, OBS_CHARS, APL_REF = 20, 20, 5
B, SEED = 5000, 0
KNOWN_FIELDS = {"uuid", "fault_category", "fault_type", "instance_type", "service", "instance", "start_time", "end_time",
                "key_observations", "key_metrics", "fault_description"}
NETWORK_FIELDS = ("source", "destination", "source_service", "destination_service", "src", "dst")


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _names(v) -> list:
    if v is None:
        return []
    return [str(x).strip() for x in (v if isinstance(v, (list, tuple)) else [v]) if str(x).strip()]


def _norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", str(s or "").lower())).strip()


def is_network(gt: dict) -> bool:
    return "network" in _norm(gt.get("fault_category"))


def targets(gt: dict) -> list:
    """Components that count as a localisation hit (R1)."""
    out = _names(gt.get("instance"))
    if is_network(gt):
        for k in NETWORK_FIELDS:
            out += _names(gt.get(k))
    return list(dict.fromkeys(out))


def la_hit(pred: dict, gt: dict) -> int:
    comp = str(pred.get("component") or "").strip()
    return int(bool(comp) and comp in targets(gt))                       # R2


def ta_hit(pred: dict, gt: dict) -> int:
    reason = _norm(" ".join(str(pred.get("reason") or "").split()[:REASON_WORDS]))
    return int(any(d and d in reason for d in (_norm(x) for x in _names(gt.get("fault_description")))))   # R3


def evidence_points(gt: dict) -> list:
    """[(type, [keywords])]: every key observation, plus each key metric not listed in a metric-type key observation (R4)."""
    pts, seen = [], set()
    for o in gt.get("key_observations") or []:
        kws = _names(o.get("keyword") if isinstance(o, dict) else o)
        typ = str(o.get("type") or "").strip().lower() if isinstance(o, dict) else ""
        if kws:
            pts.append((typ, kws))
            if typ == "metric":
                seen.update(kws)
    pts += [("metric", [m]) for m in _names(gt.get("key_metrics")) if m not in seen]
    return pts


def _observations(pred: dict) -> list:
    out = []
    for s in pred.get("reasoning_trace") or []:
        out.append(str((s.get("observation") if isinstance(s, dict) else s) or "")[:OBS_CHARS])
    return out


def point_hit(typ: str, kws: list, obs: list) -> bool:
    if typ == "metric":                                                  # strict metric-name matching: whole tokens
        toks = {t for o in obs for t in re.split(r"[^A-Za-z0-9_.]+", o) if t}
        return any(k in toks for k in kws)
    low = [o.lower() for o in obs]
    return any(k.lower() in o for k in kws for o in low)


def score_case(pred: dict, gt: dict) -> dict:
    pts, obs = evidence_points(gt), _observations(pred)
    hit = sum(point_hit(t, k, obs) for t, k in pts)
    return {"case_id": gt.get("uuid"), "fault_category": gt.get("fault_category"), "fault_type": gt.get("fault_type"),
            "instance_type": gt.get("instance_type"), "gt_component": "|".join(targets(gt)),
            "pred_component": str(pred.get("component") or ""), "la": la_hit(pred, gt), "ta": ta_hit(pred, gt),
            "exp_hit": hit, "exp_total": len(pts), "exp": hit / len(pts) if pts else float("nan"),
            "trace_len": len(pred.get("reasoning_trace") or [])}


def aggregate(d: pd.DataFrame) -> dict:
    """Eq. (1)-(4) on a table of scored cases."""
    ok = d[d["la"] == 1]
    apl = float(ok["trace_len"].mean()) if len(ok) else float("nan")
    eff = min(1.0, math.exp(-(apl - APL_REF) / APL_REF)) if len(ok) else 0.0            # R5
    la, ta = float(d["la"].mean()), float(d["ta"].mean())
    exp = float(d["exp"].mean()) if d["exp"].notna().any() else 0.0                      # R4: mean of per-case ratios
    pooled = float(d["exp_hit"].sum() / d["exp_total"].sum()) if d["exp_total"].sum() else 0.0
    return {"n": int(len(d)), "LA": la, "TA": ta, "Exp": exp, "Exp_pooled": pooled, "Eff": eff, "APL": apl,
            "Final": 100.0 * (W_LA * la + W_TA * ta + W_EXP * exp + W_EFF * eff)}


def bootstrap(d: pd.DataFrame, b: int = B, seed: int = SEED) -> tuple:
    rng, n = np.random.default_rng(seed), len(d)
    la, ta, ex, tl = (d[c].to_numpy(float) for c in ("la", "ta", "exp", "trace_len"))
    out = np.empty(b)
    for i in range(b):
        j = rng.integers(0, n, n)
        ok = la[j] == 1
        eff = min(1.0, math.exp(-(tl[j][ok].mean() - APL_REF) / APL_REF)) if ok.any() else 0.0
        e = ex[j]
        e = float(np.nanmean(e)) if np.isfinite(e).any() else 0.0
        out[i] = 100.0 * (W_LA * la[j].mean() + W_TA * ta[j].mean() + W_EXP * e + W_EFF * eff)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def score(preds: list, gts: list, partial: bool = False) -> tuple:
    """(per-case table, summary). Raises when the run does not answer every ground-truth case, unless partial (R7)."""
    by = {}
    for p in preds:
        by.setdefault(str(p.get("uuid") or p.get("case_id") or ""), p)
    missing = [g.get("uuid") for g in gts if str(g.get("uuid")) not in by]
    if missing and not partial:
        raise SystemExit(f"{len(missing)} of {len(gts)} ground-truth cases have no answer (first: {missing[:3]}). Finish the run "
                         "before scoring; --partial scores the answered cases only (debugging, not a result).")
    rows = [score_case(by[str(g.get("uuid"))], g) for g in gts if str(g.get("uuid")) in by]
    d = pd.DataFrame(rows)
    s = aggregate(d)
    s["Final_ci95"] = bootstrap(d)
    s["partial"] = bool(missing)
    s["n_missing"] = len(missing)
    d["final_case"] = 100.0 * (W_LA * d["la"] + W_TA * d["ta"] + W_EXP * d["exp"].fillna(0.0) + W_EFF * s["Eff"])   # R6
    s["by_category"] = {str(k): aggregate(g) for k, g in d.groupby("fault_category", dropna=False)}
    s["by_level"] = {str(k): aggregate(g) for k, g in d.groupby("instance_type", dropna=False)}
    return d, s


def schema_report(gts: list) -> str:
    """Field NAMES per fault category (no values), to check reading R1."""
    keys = defaultdict(Counter)
    for g in gts:
        keys[str(g.get("fault_category"))].update(g.keys())
    lines = ["Ground-truth fields per fault category (names only):"]
    for cat in sorted(keys):
        extra = sorted(set(keys[cat]) - KNOWN_FIELDS)
        lines.append(f"  {cat}: {sum(1 for g in gts if str(g.get('fault_category')) == cat)} cases"
                     + (f"; fields beyond the published example: {', '.join(extra)}" if extra else ""))
    net = [g for g in gts if is_network(g)]
    if net and not any(_names(g.get(k)) for g in net for k in NETWORK_FIELDS):
        lines.append("  WARNING: network cases carry none of the fields " + ", ".join(NETWORK_FIELDS) + "; LA of network attacks "
                     "is matched on `instance` only. Check the field names above and extend NETWORK_FIELDS before reporting.")
    return "\n".join(lines)


def _fmt(s: dict) -> str:
    lo, hi = s["Final_ci95"]
    return (f"n={s['n']}  LA={s['LA']:.4f}  TA={s['TA']:.4f} (lower bound)  Exp={s['Exp']:.4f} (pooled {s['Exp_pooled']:.4f})  "
            f"Eff={s['Eff']:.4f} (APL {s['APL']:.2f})  Final={s['Final']:.2f} [95% CI {lo:.2f}, {hi:.2f}]")


def main():
    ap = argparse.ArgumentParser(description="AIOps2025 scorer (Cai et al. 2026, eq. 1-4)")
    ap.add_argument("--pred", action="append", required=True, help="[name=]answers.jsonl (repeat for several arms)")
    ap.add_argument("--gt", required=True, help="groundtruth.jsonl")
    ap.add_argument("--out", default=None, help="folder for <name>_scored.csv (contains ground truth: private) and scores.json")
    ap.add_argument("--partial", action="store_true", help="score an unfinished run (debugging only; not a result)")
    a = ap.parse_args()
    gts = read_jsonl(a.gt)
    print(f"{len(gts)} ground-truth cases")
    print(schema_report(gts))
    tables, summaries = {}, {}
    for item in a.pred:
        name, _, path = item.rpartition("=")
        name = name or os.path.splitext(os.path.basename(path))[0]
        assert path.endswith(".jsonl"), "predictions must be the answer .jsonl (the csv has no reasoning trace)"
        tables[name], summaries[name] = score(read_jsonl(path), gts, a.partial)
        s = summaries[name]
        print(f"\n== {name}" + ("  [PARTIAL: not a result]" if s["partial"] else ""))
        print(_fmt({**s}))
        for title, key in (("by fault category", "by_category"), ("by injection level", "by_level")):
            print(f"  {title}:")
            for k, g in s[key].items():
                print(f"    {k:<24} n={g['n']:<4} LA={g['LA']:.3f} TA={g['TA']:.3f} Exp={g['Exp']:.3f} Final={g['Final']:.1f}")
    names = list(tables)
    if len(names) > 1:                                    # paired comparisons on per-case Final (R6)
        from scipy.stats import wilcoxon
        print("\nPaired differences of per-case Final (first - second; two-sided Wilcoxon):")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                m = tables[x].merge(tables[y], on="case_id", suffixes=("_x", "_y"))
                diff = (m["final_case_x"] - m["final_case_y"]).to_numpy(float)
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                print(f"  {x} - {y}: {diff.mean():+.2f}  (better {int((diff > 0).sum())}, worse {int((diff < 0).sum())}, "
                      f"equal {int((diff == 0).sum())}; p = {p:.3g})")
                summaries[f"{x}__minus__{y}"] = {"mean_diff": float(diff.mean()), "p_wilcoxon_two_sided": float(p), "n": int(len(m))}
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for name, d in tables.items():
            d.to_csv(os.path.join(a.out, f"{name}_scored.csv"), index=False)
        json.dump(summaries, open(os.path.join(a.out, "scores.json"), "w"), indent=1, default=str)
        print(f"\nwritten to {a.out} (the *_scored.csv files contain ground truth: keep them private)")


if __name__ == "__main__":
    sys.exit(main())
