#!/usr/bin/env bash
set -euo pipefail

# 0) install package (editable)
python -m pip install -e .

# 1) build engineered features (uses your CLI)
python -m esgwater.cli features

# 2) (assumes universe_filtered.parquet exists from your earlier step)
# fundamentals (resumable)
python scripts/fundamentals_yf_fast.py

# 3) forward returns (resilient batch)
python scripts/forward_returns_12m.py

# 4) build triple-adjusted labels (sector + value + profitability)
python scripts/build_labels_triple.py

# 5) train, predict, rank
python -m esgwater.cli train
python -m esgwater.cli predict
python -m esgwater.cli rank

# 6) make enriched leaderboard
python - <<'PY'
import pandas as pd
r = pd.read_csv("data/processed/rankings_ml.csv")
eng = pd.read_parquet("data/processed/engineered_water_features.parquet")
out = r.merge(eng[["symbol","sector","w_stress_weighted","pct_ops_high_stress","water_vulnerability"]],
              on="symbol", how="left").sort_values("ml_score_raw", ascending=False)
out.to_csv("data/processed/leaderboard_water_enriched.csv", index=False)
print("✅ wrote data/processed/leaderboard_water_enriched.csv")
PY

# 7) stage deliverables
mkdir -p deliverable
cp data/processed/leaderboard_water_enriched.csv \
   data/processed/rankings_ml_percentile.csv \
   data/processed/train_meta.json \
   data/processed/target_triple_adjusted_summary.txt \
   deliverable/
echo "📦 deliverable ready in ./deliverable"
