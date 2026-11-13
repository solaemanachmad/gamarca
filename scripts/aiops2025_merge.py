#!/usr/bin/env python3
"""aiops2025_merge.py -- merges parallel shard outputs (CSV, JSONL) into a unified dataset.

Usage:
    python scripts/aiops2025_merge.py --inputs "results/shards/*.jsonl" --out results/aiops2025/core
    python scripts/aiops2025_merge.py --inputs results/shards/ --out results/aiops2025/core

Scoring is a separate step, run once on the complete merged answers: scripts/aiops2025_score.py (the only code that opens the
ground truth).
"""
import argparse
import glob
import json
import os
import sys
from typing import List

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def load_jsonl_shards(patterns: List[str]) -> List[dict]:
    """Reads all JSONL records matching pattern and deduplicates by uuid."""
    files = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(glob.glob(os.path.join(pat, "**", "*.jsonl"), recursive=True))
        else:
            files.extend(glob.glob(pat))
    files = [f for f in files if f.endswith(".jsonl")]
    files = sorted(set(files))
    if not files:
        print(f"Warning: No JSONL files found matching: {patterns}", file=sys.stderr)
        return []

    records = {}
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                uuid = item.get("uuid")
                if uuid and (uuid not in records or item.get("component") or not records[uuid].get("component")):
                    records[uuid] = item                       # a non-empty answer replaces an earlier one; the last one wins
    print(f"Loaded {len(records)} unique JSONL records from {len(files)} files.")
    return list(records.values())


def load_csv_shards(patterns: List[str]) -> pd.DataFrame:
    """Reads all CSV shard tables and deduplicates by case_id."""
    files = []
    for pat in patterns:
        if os.path.isdir(pat):
            files.extend(glob.glob(os.path.join(pat, "**", "*.csv"), recursive=True))
        else:
            files.extend(glob.glob(pat))
    # Exclude scored or check files
    files = [f for f in sorted(set(files)) if f.endswith(".csv") and not f.endswith(("_scored.csv", "_check.csv", "_diagnostic.csv"))]
    if not files:
        print(f"Warning: No CSV files found matching: {patterns}", file=sys.stderr)
        return pd.DataFrame()

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)
    if "case_id" in merged.columns:
        if "error" in merged.columns:                          # a row without error is preferred over an errored one
            merged = merged.assign(_bad=merged["error"].notna()).sort_values("_bad", kind="stable").drop(columns="_bad")
        merged = merged.drop_duplicates(subset=["case_id"], keep="first")
    print(f"Loaded {len(merged)} unique CSV rows from {len(files)} files.")
    return merged


def jsonl_to_rows(records: List[dict]) -> List[dict]:
    """Converts JSONL records into tabular summary rows when CSV shards are missing."""
    rows = []
    for r in records:
        reason = r.get("reason", "")
        main_part = reason.split(";")[0].strip() if ";" in reason else reason
        fault_type = main_part.split(" at ")[0].strip() if " at " in main_part else main_part
        rows.append({
            "case_id": r.get("uuid", ""),
            "component": r.get("component", ""),
            "fault_type": fault_type,
            "confidence": 1.0,
            "reason": reason,
            "n_steps": len(r.get("reasoning_trace", [])),
            "time": r.get("time", ""),
        })
    return rows


def expand_zips(patterns: List[str], tmp: str) -> List[str]:
    """Shard outputs downloaded as *_all.zip (one per session): extract each into its own folder under tmp and use the folders."""
    import zipfile
    out = []
    for pat in patterns:
        hits = [h for h in (glob.glob(pat) or [pat]) if h.endswith(".zip") and os.path.isfile(h)]
        if not hits:
            out.append(pat)
            continue
        for h in sorted(hits):
            d = os.path.join(tmp, f"z{len(out)}")
            with zipfile.ZipFile(h) as z:
                z.extractall(d)
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(description="Merge parallel AIOps2025 shards")
    ap.add_argument("--inputs", nargs="+", required=True, help="paths, globs, directories or the *_all.zip files of the shard sessions")
    ap.add_argument("--out", default="results/aiops2025/core", help="output directory for merged results")
    ap.add_argument("--name", default="summary", help="base output name (default: summary)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="aiops_merge_")
    a.inputs = expand_zips(a.inputs, tmp)

    jsonl_records = load_jsonl_shards(a.inputs)
    csv_df = load_csv_shards(a.inputs)
    if csv_df.empty and jsonl_records:
        csv_df = pd.DataFrame(jsonl_to_rows(jsonl_records))

    out_jsonl = os.path.join(a.out, f"{a.name}.jsonl")
    out_csv = os.path.join(a.out, f"{a.name}.csv")

    if jsonl_records:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for item in jsonl_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved merged JSONL -> {out_jsonl}")

    if not csv_df.empty:
        csv_df.to_csv(out_csv, index=False)
        print(f"Saved merged CSV   -> {out_csv}")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()