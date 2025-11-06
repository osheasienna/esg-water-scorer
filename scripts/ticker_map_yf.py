import time
from pathlib import Path
import pandas as pd, yfinance as yf

UNIV = Path("data/processed/universe_filtered.parquet")
OUT  = Path("data/processed/symbol_map.csv")

# Common Yahoo suffixes by exchange; try original + these
SUFFIXES = ["", ".TO", ".V", ".L", ".PA", ".AS", ".BR", ".MC", ".MI", ".DE",
            ".SW", ".VI", ".CO", ".OL", ".ST", ".HE", ".DK", ".SI", ".HK",
            ".NS", ".BO", ".AX", ".NZ", ".SA", ".MX"]

u = pd.read_parquet(UNIV)
syms = u["symbol"].dropna().astype(str).unique().tolist()

rows = []
for i, s in enumerate(syms, 1):
    found = None
    # If already has a dot, try as-is first
    try_first = [s] + [s.split(".")[0] + suf for suf in SUFFIXES if suf and not s.endswith(suf)]
    tried = set()
    for cand in try_first:
        if cand in tried: 
            continue
        tried.add(cand)
        try:
            t = yf.Ticker(cand)
            df = t.history(period="3y", interval="1d", auto_adjust=True)
            if df is not None and not df.empty and "Close" in df:
                found = cand
                break
        except Exception:
            pass
        time.sleep(0.02)
    rows.append({"symbol": s, "symbol_yf": found})
    if i % 25 == 0:
        print(f"[{i}/{len(syms)}] mapped so far…", flush=True)

m = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(OUT, index=False)
print(f"✅ wrote {OUT} — mapped {m['symbol_yf'].notna().sum()} / {len(m)}")
