
from __future__ import annotations
import pandas as pd
from pathlib import Path
from esgwater.features.water_ops import (
    load_facilities, load_aqueduct_lookup, facility_water_stress,
    load_sector_vuln, attach_sector_vulnerability
)

def build_engineered_features() -> str:
    # 1) load base universe
    uni = pd.read_parquet("data/processed/universe_latest.parquet")

    # 2) bring in sector from manual_features.csv if present
    mf = Path("data/raw/manual_features.csv")
    if mf.exists():
        try:
            mfd = pd.read_csv(mf)
            cols = [c for c in ["symbol","sector"] if c in mfd.columns]
            if cols:
                uni = uni.merge(mfd[cols], on="symbol", how="left")
        except Exception:
            pass

    # 3) sector vulnerability map
    secmap = load_sector_vuln()
    uni2 = attach_sector_vulnerability(uni, secmap)

    # 4) facilities -> water stress per symbol
    fac = load_facilities()
    aq = load_aqueduct_lookup()
    stress = facility_water_stress(fac, aq)  # returns: symbol, w_stress_weighted, pct_ops_high_stress

    # 5) merge & defaults
    out = uni2.merge(stress, on="symbol", how="left")
    out["w_stress_weighted"] = out["w_stress_weighted"].fillna(2.5)
    out["pct_ops_high_stress"] = out["pct_ops_high_stress"].fillna(0.0)
    out["water_vulnerability"] = out["water_vulnerability"].fillna(0.3)

    # 6) write parquet
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    cols = ["symbol","sector","water_vulnerability","w_stress_weighted","pct_ops_high_stress"]
    out[cols].to_parquet("data/processed/engineered_water_features.parquet", index=False)
    return "data/processed/engineered_water_features.parquet"
