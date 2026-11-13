# AIOps2025 Benchmark Results

This directory contains evaluation outputs, predictions, and empirical scorecards for **GAMA-RCA** transferred zero-shot to the **AIOps2025** benchmark (CCF International AIOps Challenge; HipsterShop + TiDB + Redis on Kubernetes, Chaos Mesh faults, 400 cases; Cai et al., 2026).

---

## 🔬 Benchmark Overview

* **Zero-Shot Transfer**: Evaluates generalisation to an unseen microservice topology without retraining or tuning any statistical hyperparameters.
* **Evaluation Metrics**:
  * **Location Accuracy (LA)**: the predicted component string-matches the ground truth exactly (pod-level faults need the exact pod; network attacks accept source or destination).
  * **Type Accuracy (TA)**: the first 20 words of the reason contain one of the ground-truth fault descriptions (keyword part only: a lower bound).
  * **Explainability (Exp)**: share of the ground-truth evidence points found in the first 20 characters of the trace observations.
  * **Efficiency (Eff)**: $\min(1, e^{-(APL-5)/5})$, APL = mean trace length over the LA-correct cases.
  * **Final Score**: $(0.4\,\text{LA} + 0.4\,\text{TA} + 0.1\,\text{Exp} + 0.1\,\text{Eff}) \times 100$ (Cai et al. 2026, eq. 1-4; re-implemented in `scripts/aiops2025_score.py`, pre-registered in `PREREGISTRATION_AIOPS2025.md`).

---

## 📁 Directory Structure

```
results/aiops2025/
├── README.md                      # Benchmark instructions & guidelines
└── core/                          # Outputs from GAMA Core (CPU Only)
    ├── README.md                  # Output format specification
    ├── summary.csv                # 12-column per-case diagnosis summary
    ├── predictions.jsonl          # Submission JSONL format with step-anchored trace
    └── (scores are written by scripts/aiops2025_score.py; *_scored.csv contains ground truth and is git-ignored)
```

---

## ⚡ Running on CPU (parallel, resumable)

### 1. One command, several processes (Kaggle CPU session: 4 vCPUs)
```bash
python scripts/aiops2025_run.py --data /kaggle/input/aiops2025 --out results/aiops2025/core --workers 4
```
`--workers n` starts n independent shard processes (contiguous blocks of the cases sorted by window start; neighbouring cases share
files, so the file cache is reused). Each shard writes its own files under `<out>/<name>_shards/` after every case. A shard that dies
(out of memory, killed session) is restarted up to `--retries` times and continues where it stopped; at the end the shards are merged
into `<out>/AIOPS_GAMA_core.{jsonl,csv}`. The partition does not change any result. If memory is short: `AIOPS_CACHE_FILES=8` and fewer
workers; the per-case peak memory is in the `peak_rss_mb` column.

If the session ends, run **the same command again**: finished cases are kept, the rest are run.

### 2. Four sessions (one quarter each)
In each of four Kaggle sessions / accounts run one shard (k = 0, 1, 2, 3):
```bash
python scripts/aiops2025_run.py --data /kaggle/input/aiops2025 --out /kaggle/working/out --shard 0/4
```
Each session ends with `/kaggle/working/out/AIOPS_GAMA_core_shard0of4_all.zip` (answers, table, manifest). Download the four zips and merge
them (the zips are read directly):
```bash
python scripts/aiops2025_merge.py --inputs AIOPS_GAMA_core_shard*of4_all.zip --out results/aiops2025/core --name AIOPS_GAMA_core
```
LLM arms use one process per shard (`--shard k/n`, not `--workers`): the API has a rate limit.

### Saving the output on Kaggle
Files in `/kaggle/working` are kept only for a notebook version that finished: use **Save Version → Save & Run All (Commit)** and download
the zip from that version's Output tab (an interactive session loses them when it stops). The zip is rewritten every 20 cases
(`--workers`: every minute), so a stopped run still leaves what it had done; run the same command again in a new session to continue
(copy the zip back, unzip it into `--out`, then run the command: the finished cases are kept).

---

### 3. Scoring (once, after every arm has answered all 400 cases)
The scorer is the only code that opens the ground truth; it refuses an unfinished run.
```bash
python scripts/aiops2025_score.py \
  --pred core=results/aiops2025/core/AIOPS_GAMA_core.jsonl \
  --pred hybrid=results/aiops2025/hybrid/AIOPS_GAMA_hybrid_gptoss120b_cerebras.jsonl \
  --pred evidence=results/aiops2025/evidence/AIOPS_EVIDENCE_LLM_gptoss120b_cerebras.jsonl \
  --gt /kaggle/input/aiops2025/groundtruth.jsonl --out results/aiops2025/scores
```

---

## 📋 CSV Format

`<name>.csv` has one row per case: `case_id`, `component`, `fault_type`, `confidence` (posterior P(e*)), `reason`, `n_steps`, `n_metrics`,
`n_logs`, `n_traces`, `load_time_s`, `pipeline_time_s`, `wall_s`, `peak_rss_mb`; a case that errored has `error` set and no answer line.
