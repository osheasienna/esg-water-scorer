import os, time
import pandas as pd
from pathlib import Path

# yfinance can be slow; disable websockets for stability
os.environ.setdefault("YF_USE_WEBSOCKET", "False")
import yfinance as yf

# ---------------- config ----------------
BATCH = int(os.getenv("YF_BATCH", "40"))       # symbols per batch
SLEEP = float(os.getenv("YF_SLEEP", "0.15"))   # pause between batches
OUT   = Path("data/raw/fundamentals.csv")
UNIV  = Path("data/processed/universe_filtered.parquet")
# ----------------------------------------

u = pd.read_parquet(UNIV)
syms = u["symbol"].dropna().astype(str).unique().tolist()

# resume if file exists
if OUT.exists():
    try:
        done = set(pd.read_csv(OUT)["symbol"].astype(str))
    except Exception:
        done = set()
else:
    done = set()

def log(msg): print(msg, flush=True)

rows = []
total = len(syms)
log(f"Starting fundamentals: {total} symbols; resume has {len(done)}.")

for i in range(0, total, BATCH):
    chunk = [s for s in syms[i:i+BATCH] if s not in done]
    if not chunk:
        continue

    log(f"[{i+1:>4}/{total}] Fetching {len(chunk)} symbols...")
    tk = yf.Tickers(" ".join(chunk))

    for s in chunk:
        try:
            t = tk.tickers.get(s)
            if not t:
                continue
            info = t.get_info()  # may be slow; we show progress + save per batch
            rows.append({
                "symbol": s,
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "profit_margin": info.get("profitMargins"),
            })
        except Exception:
            continue

    # append + write incrementally
    if rows:
        df_new = pd.DataFrame(rows)
        rows.clear()
        for c in ["pb", "roe", "profit_margin"]:
            df_new[c] = pd.to_numeric(df_new[c], errors="coerce")

        if OUT.exists():
            df_old = pd.read_csv(OUT)
            df_all = pd.concat([df_old, df_new], ignore_index=True)\
                       .drop_duplicates(subset=["symbol"], keep="last")
        else:
            df_all = df_new.drop_duplicates(subset=["symbol"])

        OUT.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(OUT, index=False)
        done = set(df_all["symbol"].astype(str))
        log(f"   ↳ saved: {len(df_new)} new | total: {len(df_all)}")

    time.sleep(SLEEP)

log("✅ Done. Fundamentals written to data/raw/fundamentals.csv")
