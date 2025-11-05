import os, time, requests, pandas as pd
from pathlib import Path

API = os.getenv("ALPHAVANTAGE_API_KEY")
if not API:
    raise SystemExit("Set ALPHAVANTAGE_API_KEY first (export ALPHAVANTAGE_API_KEY=\"…\").")

BATCH = int(os.getenv("AV_BATCH", "60"))  # adjust if you want more/fewer each run

uni = pd.read_parquet("data/processed/universe_latest.parquet")
symbols = uni["symbol"].dropna().astype(str).unique().tolist()[:BATCH]

mf_p = Path("data/raw/manual_features.csv"); mf_p.parent.mkdir(parents=True, exist_ok=True)
mf = pd.read_csv(mf_p) if mf_p.exists() else pd.DataFrame(columns=["symbol","sector","country"])

def fetch_overview(sym: str):
    r = requests.get("https://www.alphavantage.co/query",
                     params={"function": "OVERVIEW", "symbol": sym, "apikey": API},
                     timeout=30)
    j = r.json()
    sec = (j.get("Sector") or "").strip()
    ctry = (j.get("Country") or "").strip().upper()
    return sec, ctry

rows, seen = [], set(mf["symbol"].astype(str)) if not mf.empty else set()
for s in symbols:
    if s in seen:
        continue
    try:
        sec, ctry = fetch_overview(s)
        if not sec and not ctry:
            continue
        rows.append({"symbol": s, "sector": sec, "country": ctry})
    except Exception:
        continue
    time.sleep(12)

if rows:
    mf = pd.concat([mf, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset=["symbol"])
    mf.to_csv(mf_p, index=False)
    print(f"✅ manual_features.csv +{len(rows)} rows")
else:
    print("ℹ️ no new manual features")

# facilities
fac_p = Path("data/raw/facilities.csv")
fac = pd.read_csv(fac_p) if fac_p.exists() else pd.DataFrame(columns=["symbol","site_id","country","region","lat","lon","weight"])
have = set(fac["symbol"].astype(str)) if not fac.empty else set()
mf2 = mf.set_index("symbol")

new_fac = []
for s in symbols:
    if s in have or s not in mf2.index:
        continue
    ctry = mf2.at[s, "country"]
    if isinstance(ctry, str) and ctry:
        new_fac.append({
            "symbol": s, "site_id": f"{s}_ANY_1", "country": ctry, "region": "ANY",
            "lat": "", "lon": "", "weight": 1.0
        })

if new_fac:
    fac = pd.concat([fac, pd.DataFrame(new_fac)], ignore_index=True).drop_duplicates(subset=["symbol","site_id"])
    fac.to_csv(fac_p, index=False)
    print(f"✅ facilities.csv +{len(new_fac)} country rows")
else:
    print("ℹ️ no new facilities")

print("Done.")