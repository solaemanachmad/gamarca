# AIOps2025 GAMA Core Results

This subfolder stores the run outputs for **GAMA Core (CPU Only)** on AIOps2025.

### Output Files
* `summary.csv`: The 12-column per-case diagnosis results.
* `predictions.jsonl`: Benchmark submission JSONL lines containing component, reason, and reasoning trace steps.
* Scores: `scripts/aiops2025_score.py`, run once on the complete answers (`*_scored.csv` contains ground truth and is git-ignored).

### Execution Command
```bash
python scripts/aiops2025_run.py \
  --data /kaggle/input/aiops2025 \
  --out results/aiops2025/core \
  --workers 4
```
