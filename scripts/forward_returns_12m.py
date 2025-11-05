from pathlib import Path
import time
import pandas as pd
import numpy as np
import yfinance as yf

UNIV = Path("data/processed/universe_filtered.parquet")
OUT  = Path("data/processed/forward_returns_12m.parquet")

def fwd_12m_ret(px: pd.Series, h: int = 252) -> pd.Series:
    return px.shift(-h) / px - 1.0

u = pd.read_parquet(UNIV)
syms = u["symbol"].dropna().astype(str).unique().tolist()

# download via Ticker().history (more reliable than yf.download for some tickers)
rows = []
ok = 0; bad = 0
for i, s in enumerate(syms, 1):
    for attempt in (1, 2):
        try:
            t = yf.Ticker(s)
            df = t.history(period="5y", interval="1d", auto_adjust=True)
            if df is None or df.empty or "Close" not in df.columns:
                raise RuntimeError("empty")
            px = df["Close"]
            r12 = fwd_12m_ret(px)
            tmp = pd.DataFrame({"symbol": s, "date": px.index, "ret_fwd_12m": r12.values})
            rows.append(tmp)
            ok += 1
            break
        except Exception:
            if attempt == 2:
                bad += 1
            time.sleep(0.2)
    if i % 25 == 0:
        print(f"[{i}/{len(syms)}] ok={ok} bad={bad}", flush=True)
    time.sleep(0.05)

if not rows:
    raise SystemExit("No price data downloaded for any symbols (still empty).")

allr = pd.concat(rows, ignore_index=True).dropna(subset=["ret_fwd_12m"])

# choose one anchor per symbol: last date where forward return is defined
idx = (allr.sort_values(["symbol","date"])
            .groupby("symbol", as_index=False)
            .tail(1))

OUT.parent.mkdir(parents=True, exist_ok=True)
idx.to_parquet(OUT, index=False)
print(f"✅ wrote {OUT} for {idx['symbol'].nunique()} symbols (ok={ok}, bad={bad})")
