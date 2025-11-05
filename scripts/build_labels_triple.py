from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

UNIV = Path("data/processed/universe_filtered.parquet")
RET  = Path("data/processed/forward_returns_12m.parquet")
FUND = Path("data/raw/fundamentals.csv")
LBL  = Path("data/raw/labels.csv")
SUM  = Path("data/processed/target_triple_adjusted_summary.txt")

u  = pd.read_parquet(UNIV)
fr = pd.read_parquet(RET)
fd = pd.read_csv(FUND)

# Merge
df = (fr.merge(u, on="symbol", how="inner")
        .merge(fd, on="symbol", how="left"))

# Keep only rows with a valid forward return
df = df[pd.to_numeric(df["ret_fwd_12m"], errors="coerce").replace([np.inf, -np.inf], np.nan).notna()]

# --- Value & Profitability proxies ---
# P/B must be positive to take log
pb = pd.to_numeric(df["pb"], errors="coerce")
pb = pb.where(pb > 0)  # set nonpositive to NaN
df["bm"] = -np.log(pb)  # higher bm => cheaper

# Profitability: ROE, fallback to profit_margin
roe = pd.to_numeric(df["roe"], errors="coerce")
pm  = pd.to_numeric(df["profit_margin"], errors="coerce")
prof = roe.where(roe.notna(), pm)
df["prof"] = prof

# Standardize (z-score) where possible
for c in ["bm","prof"]:
    x = pd.to_numeric(df[c], errors="coerce")
    mu, sd = x.mean(skipna=True), x.std(skipna=True)
    df[c] = ((x - mu) / sd) if pd.notna(sd) and sd > 0 else 0.0

# --- Design matrix X: const + bm + prof + sector dummies ---
sec_dum = pd.get_dummies(df["sector"].astype(str), prefix="sec", drop_first=True)
X = pd.concat([df[["bm","prof"]].apply(pd.to_numeric, errors="coerce"), sec_dum], axis=1)

# Coerce everything numeric, drop any rows with NaNs in X or y
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
y = pd.to_numeric(df["ret_fwd_12m"], errors="coerce").replace([np.inf, -np.inf], np.nan)

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]
syms = df.loc[mask, "symbol"].astype(str).values

# Add constant and fit
X = sm.add_constant(X, has_constant="add")
model = sm.OLS(y.values.astype(float), X.values.astype(float)).fit()

# Residual target
resid = pd.Series(model.resid, index=y.index)
lab = pd.DataFrame({"symbol": syms, "target": resid.values})

# Save
LBL.parent.mkdir(parents=True, exist_ok=True)
lab.to_csv(LBL, index=False)

SUM.parent.mkdir(parents=True, exist_ok=True)
with open(SUM, "w") as f:
    f.write(model.summary().as_text())

print(f"✅ labels.csv written for {len(lab)} symbols (kept {len(lab)} rows for OLS)")
