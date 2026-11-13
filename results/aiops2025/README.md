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

## ⚡ Running on CPU (Sequential & Parallel)

### 1. Single Command Multi-Core Execution (Fastest on Kaggle)
Kaggle CPU notebooks provide 4 vCPUs. To utilize all 4 CPU cores simultaneously:
```bash
python scripts/aiops2025_run.py \
  --data /kaggle/input/aiops2025 \
  --out results/aiops2025/core \
  --workers 4
```

### 2. Parallel Sharding & Merging (Distributed / Multi-session)
If running multiple background shards:
```bash
# Run 4 parallel shards
for k in 0 1 2 3; do
  python scripts/aiops2025_run.py \
    --data /kaggle/input/aiops2025 \
    --out /kaggle/working/shards \
    --shard ${k}/4 &
done
wait

# Merge all shards
python scripts/aiops2025_merge.py \
  --inputs "/kaggle/working/shards/*.jsonl" \
  --out results/aiops2025/core
```

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

## 📋 CSV Format (12 Clean Columns)

The output `summary.csv` provides clean, tabular diagnostics:
1. `case_id`: Incident identifier (UUID)
2. `component`: Identified root cause entity (pod / service / node)
3. `fault_type`: Identified fault mechanism (e.g. `cpu_stress`, `network_delay`)
4. `confidence`: Posterior Bayesian confidence $P(e^*)$
5. `reason`: Concise summary of the root cause and holding evidence
6. `n_steps`: Number of causal reasoning steps in the evidence chain
7. `n_metrics`: Total metric time-series points evaluated
8. `n_logs`: Total container log lines evaluated
9. `n_traces`: Total distributed trace spans evaluated
10. `load_time_s`: Telemetry data loading duration (seconds)
11. `pipeline_time_s`: GAMA evidence calibration & Bayesian reasoning duration (seconds)
12. `wall_s`: Total execution wall-clock time per case (seconds)
