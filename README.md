# ESG-Water-Scorer
A data-driven model ranking global equities by **investability under water-scarcity risk**, integrating sustainability indicators, sectoral vulnerability, and regional water stress.  
The model produces normalized scores, percentile rankings, and enriched leaderboards.

---

## ✨ Key Features
✅ Water-scarcity modeling (regional + sectoral)  
✅ ESG + fundamental financial enrichment  
✅ Triple-adjusted 12-month forward return target  
✅ Group-KFold by sector (out-of-sample discipline)  
✅ ML-based scoring with explainability hooks  
✅ CLI pipeline for reproducible execution  

---

## 🔎 Method Overview

### 1) Universe
We construct a global equity universe from publicly listed securities with available fundamentals and geographic operational exposure.

### 2) Water Dimensions
**Regional water exposure**
- Weighted baseline water stress from Aqueduct
- % of operational footprint in high-stress regions

**Sectoral vulnerability**
- Mapping sectors to water dependency
- Semiconductor, Utilities, Materials → high exposure

### 3) ESG & Financial Indicators
Features include:
- Water Stress (weighted, squared)
- Sector vulnerability score
- ROE
- Profit margin
- Book-to-market
- Interaction: water-stress × vulnerability

All sector-demeaned (“z_sec”) to control for structural biases.

### 4) Target Definition
We forecast **12-month forward total return**, triple-adjusted:
- Sector-neutral
- Country-neutral
- Market-neutral

Converted to percentile rank → used as training target.

### 5) Modeling
Model: `RandomForestRegressor`  
Training scheme: `GroupKFold` (sector groups)

Outputs:
- `train_meta.json`
- `model.joblib`

Performance:
| Metric | Result |
|---|---|
| Cross-validated IC | ~ **0.24** |
| R² | ≈ 0 (expected — financial returns noisy) |

Interpretation: meaningful rank information; not level forecasting.

---

## 📦 Installation

```bash
git clone https://github.com/osheasienna/esg-water-scorer.git
cd esg-water-scorer
pip install -e .
