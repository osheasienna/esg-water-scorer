from pathlib import Path
import time, pandas as pd, numpy as np, yfinance as yf

UNIV = Path("data/processed/universe_filtered.parquet")
MAP  = Path("data/processed/symbol_map.csv")
OUT  = Path("data/processed/forward_returns_12m.parquet")

def fwd_12m_ret(px, h=252):
    return px.shift(-h)/px - 1.0

u   = pd.read_parquet(UNIV)
mp  = pd.read_csv(MAP) if MAP.exists() else pd.DataFrame(columns=["symbol","symbol_yf"])
dfu = u.merge(mp, on="symbol", how="left")
dfu["symbol_use"] = dfu["symbol_yf"].fillna(dfu["symbol"]).astype(str)

rows=[]; ok=0; bad=0
for i, s in enumerate(dfu["symbol_use"].unique(), 1):
    try:
        t = yf.Ticker(s)
        h = t.history(period="5y", interval="1d", auto_adjust=True)
        if h is None or h.empty or "Close" not in h: raise RuntimeError("empty")
        r12 = fwd_12m_ret(h["Close"])
        rows.append(pd.DataFrame({"symbol_yf": s, "date": h.index, "ret_fwd_12m": r12.values}))
        ok += 1
    except Exception:
        bad += 1
    if i % 25 == 0:
        print(f"[{i}/{dfu['symbol_use'].nunique()}] ok={ok} bad={bad}", flush=True)
    time.sleep(0.02)

if not rows:
    raise SystemExit("No price data downloaded.")
allr = pd.concat(rows, ignore_index=True).dropna(subset=["ret_fwd_12m"])

# One anchor per mapped symbol: last date with forward return defined
idx = (allr.sort_values(["symbol_yf","date"])
           .groupby("symbol_yf", as_index=False).tail(1))

# bring back original symbols
idx = idx.merge(dfu[["symbol","symbol_use"]].drop_duplicates(),
                left_on="symbol_yf", right_on="symbol_use", how="left").drop(columns=["symbol_use"])

OUT.parent.mkdir(parents=True, exist_ok=True)
idx.to_parquet(OUT, index=False)
print(f"✅ wrote {OUT} for {idx['symbol'].nunique()} symbols (ok={ok}, bad={bad})")
