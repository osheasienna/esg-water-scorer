
import os, re, time, pandas as pd
from pathlib import Path

# yfinance imports kept inside function to avoid heavy import if not needed
os.environ.setdefault("YF_USE_WEBSOCKET", "False")  # keep it simple/robust

BATCH = int(os.getenv("YF_BATCH", "80"))  # tune as needed

# --- helpers ---
BAD_PAT = re.compile(
    r"""(
        -P-      |   # preferreds (e.g., ALL-P-J)
        -U$      |   # SPAC units
        -WS?$    |   # warrants -W / -WS
        -WT$     |   # warrant token
        \.WT     |   # .WT suffix
        \.PR     |   # .PR preferred marker
        /        |   # slashes
        \s       |   # whitespace/specials
        ^-       |   # leading hyphen (seen in '-P-HIZ')
        \^       |   # carets in some preferred syntax
        =        |   # equals (junk)
        \$           # dollar sign (junk)
    )""",
    re.VERBOSE | re.IGNORECASE,
)

def clean_universe_symbols(universe: pd.DataFrame) -> list[str]:
    syms = universe["symbol"].dropna().astype(str).unique().tolist()
    # filter out garbage formats
    syms = [s.strip() for s in syms if s and not BAD_PAT.search(s)]
    # simple sanity: length and alphanum/hyphen/dot
    keep = []
    for s in syms:
        if 1 <= len(s) <= 12 and re.fullmatch(r"[A-Za-z0-9\.\-]+", s):
            keep.append(s)
    return keep

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def fetch_yf_batch(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf
    out_rows = []
    # Use Tickers batch object to reuse a session
    tk = yf.Tickers(" ".join(symbols))
    for s in symbols:
        try:
            t = tk.tickers.get(s)
            if t is None:
                continue
            info = t.get_info()  # yfinance >=0.2.66 lazily fetches; robust but can be slow
            sec = (info.get("sector") or "").strip()
            cty = (info.get("country") or "").strip().upper()
            if sec or cty:
                out_rows.append({"symbol": s, "sector": sec, "country": cty})
        except Exception:
            # ignore bad symbols/timeouts
            continue
        # small pause to be polite
        time.sleep(0.05)
    return pd.DataFrame(out_rows)

def main():
    uni = pd.read_parquet("data/processed/universe_latest.parquet")
    symbols = clean_universe_symbols(uni)
    if not symbols:
        print("⚠️ No symbols after cleaning.")
        return

    mf_p = Path("data/raw/manual_features.csv")
    mf_p.parent.mkdir(parents=True, exist_ok=True)
    mf = pd.read_csv(mf_p) if mf_p.exists() else pd.DataFrame(columns=["symbol","sector","country"])
    existing = set(mf["symbol"].astype(str)) if not mf.empty else set()

    added = 0
    for chunk in chunked([s for s in symbols if s not in existing], BATCH):
        df = fetch_yf_batch(chunk)
        if not df.empty:
            mf = pd.concat([mf, df], ignore_index=True).drop_duplicates(subset=["symbol"])
            added += len(df)
            # write incrementally so you can stop/restart
            mf.to_csv(mf_p, index=False)
            print(f"✅ added {len(df)} (total now {len(mf)})")
        else:
            print("ℹ️ batch produced 0 rows (all not-found?)")
        # brief pause between batches
        time.sleep(0.5)

    # facilities fallback: add a country-level site for any symbol in MF w/o a site
    fac_p = Path("data/raw/facilities.csv")
    fac = pd.read_csv(fac_p) if fac_p.exists() else pd.DataFrame(
        columns=["symbol","site_id","country","region","lat","lon","weight"]
    )
    have = set(fac["symbol"].astype(str)) if not fac.empty else set()
    mf2 = mf.set_index("symbol")

    new_fac = []
    for s in mf2.index:
        if s in have:
            continue
        ctry = mf2.at[s, "country"]
        if isinstance(ctry, str) and ctry:
            new_fac.append({
                "symbol": s, "site_id": f"{s}_ANY_1", "country": ctry, "region": "ANY",
                "lat": "", "lon": "", "weight": 1.0
            })
    if new_fac:
        fac = pd.concat([fac, pd.DataFrame(new_fac)], ignore_index=True)\
                .drop_duplicates(subset=["symbol","site_id"])
        fac.to_csv(fac_p, index=False)
        print(f"✅ facilities.csv +{len(new_fac)} country-level rows")
    else:
        print("ℹ️ no new facilities needed")

    print(f"Done. Newly added from Yahoo: {added}")

if __name__ == "__main__":
    main()
