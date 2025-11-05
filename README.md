# ESG Water Scorer (MVP)

Rank equities on **investibility** using **water scarcity exposure + ESG** features and a **triple-adjusted** financial target (12-month forward returns residualized by sector, value, and profitability).

## What this does
- Builds water features:
  - `w_stress_weighted` (0–5, WRI Aqueduct baseline stress, ops-weighted)
  - `pct_ops_high_stress` (share of facilities in high-stress regions)
  - `water_vulnerability` (sector-level prior, 0–1)
- Creates **labels**: 12M forward return residuals after OLS on sector dummies, value (−log P/B), profitability (ROE/PM).
- Trains ML (GroupKFold by sector), predicts, and ranks.

## Quickstart
```bash
python -m pip install -e .
python -m esgwater.cli features        # build engineered water features
python scripts/fundamentals_yf_fast.py # fetch P/B, ROE, ProfitMargin
python scripts/forward_returns_12m.py  # 12M forward returns
python scripts/build_labels_triple.py  # residual target
python -m esgwater.cli train           # GroupKFold by sector (5)
python -m esgwater.cli predict
python -m esgwater.cli rank

