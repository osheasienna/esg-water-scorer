import os, time
import pandas as pd
from pathlib import Path
os.environ.setdefault("YF_USE_WEBSOCKET","False")
import yfinance as yf

u = pd.read_parquet("data/processed/universe_filtered.parquet")
syms = u["symbol"].dropna().astype(str).unique().tolist()

rows = []
for i in range(0, len(syms), 100):
    chunk = syms[i:i+100]
    tk = yf.Tickers(" ".join(chunk))
    for s in chunk:
        try:
            t = tk.tickers.get(s)
            if not t:
                continue
            info = t.get_info()
            rows.append({
                "symbol": s,
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "profit_margin": info.get("profitMargins"),
            })
        except Exception:
            continue
    time.sleep(0.3)

out = pd.DataFrame(rows).drop_duplicates(subset=["symbol"])
for c in ["pb","roe","profit_margin"]:
    out[c] = pd.to_numeric(out[c], errors="coerce")

Path("data/raw").mkdir(parents=True, exist_ok=True)
out.to_csv("data/raw/fundamentals.csv", index=False)
print(f"✅ fundamentals.csv rows: {len(out)}")
