
from __future__ import annotations
import pandas as pd
from pathlib import Path

# ------------------------------
# Helpers
# ------------------------------
def _country_iso2(x: str) -> str:
    if not isinstance(x, str): return ""
    s = x.strip().upper()
    m = {
        "UNITED STATES":"US","UNITED STATES OF AMERICA":"US","USA":"US","U.S.":"US",
        "UNITED KINGDOM":"GB","GREAT BRITAIN":"GB","UK":"GB",
        "SOUTH KOREA":"KR","KOREA, REPUBLIC OF":"KR","KOREA":"KR",
        "NORTH KOREA":"KP","RUSSIA":"RU","RUSSIAN FEDERATION":"RU",
        "CZECH REPUBLIC":"CZ","VIET NAM":"VN","VIETNAM":"VN",
        "TAIWAN":"TW","REPUBLIC OF CHINA":"TW",
        "HONG KONG":"HK","MACAU":"MO","CHINA":"CN",
        "IVORY COAST":"CI","COTE D'IVOIRE":"CI","CÔTE D'IVOIRE":"CI",
        "TURKIYE":"TR","TURKEY":"TR","UAE":"AE","UNITED ARAB EMIRATES":"AE",
        "SAUDI ARABIA":"SA","JAPAN":"JP","CANADA":"CA","FRANCE":"FR","GERMANY":"DE",
        "SPAIN":"ES","ITALY":"IT","NETHERLANDS":"NL","BELGIUM":"BE","SWEDEN":"SE",
        "NORWAY":"NO","DENMARK":"DK","FINLAND":"FI","SWITZERLAND":"CH","AUSTRALIA":"AU",
        "NEW ZEALAND":"NZ","MEXICO":"MX","BRAZIL":"BR","ARGENTINA":"AR","CHILE":"CL",
        "SOUTH AFRICA":"ZA","INDIA":"IN","INDONESIA":"ID","MALAYSIA":"MY","SINGAPORE":"SG",
        "THAILAND":"TH","PHILIPPINES":"PH","POLAND":"PL","AUSTRIA":"AT","IRELAND":"IE",
        "PORTUGAL":"PT","GREECE":"GR","HUNGARY":"HU","ROMANIA":"RO","BULGARIA":"BG",
        "CROATIA":"HR","SLOVENIA":"SI","SLOVAKIA":"SK","LUXEMBOURG":"LU","ISRAEL":"IL",
        "EGYPT":"EG","MOROCCO":"MA","TUNISIA":"TN","COLOMBIA":"CO","PERU":"PE",
        "URUGUAY":"UY","PARAGUAY":"PY"
    }
    if len(s)==2 and s.isalpha(): return s
    return m.get(s, s[:2])

# ------------------------------
# Loaders
# ------------------------------
def load_facilities(path: str = "data/raw/facilities.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["symbol","site_id","country","region","lat","lon","weight"])
    df = pd.read_csv(p)
    for c in ["symbol","site_id","country","region"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    return df

def load_aqueduct_lookup(path: str = "data/reference/aqueduct_lookup.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["country","region","baseline_water_stress"])
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    # normalize keys and value
    df["country"] = df["country"].apply(_country_iso2)
    df["region"]  = df["region"].astype(str).str.upper().fillna("ANY")
    col = "baseline_water_stress"
    if col not in df.columns:
        for c in df.columns:
            if c.strip().lower().replace(" ","_") == "baseline_water_stress":
                df[col] = df[c]
                break
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["country","region",col]]

def load_sector_vuln(path: str = "data/reference/sector_water_vulnerability.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["sector","water_vulnerability"])
    df = pd.read_csv(p)
    if "water_vulnerability" in df.columns:
        df["water_vulnerability"] = pd.to_numeric(df["water_vulnerability"], errors="coerce")
    return df

def attach_sector_vulnerability(universe: pd.DataFrame, secmap: pd.DataFrame) -> pd.DataFrame:
    u = universe.copy()
    if "sector" not in u.columns:
        u["sector"] = pd.NA
    return u.merge(secmap, on="sector", how="left")

# ------------------------------
# Core aggregation (robust)
# ------------------------------
def facility_water_stress(facilities: pd.DataFrame, aq: pd.DataFrame) -> pd.DataFrame:
    if facilities.empty:
        return pd.DataFrame(columns=["symbol","w_stress_weighted","pct_ops_high_stress"])

    fac = facilities.copy()
    fac["country_norm"] = fac["country"].apply(_country_iso2)
    fac["region"] = fac["region"].astype(str).str.upper().fillna("ANY")

    aq2 = aq.copy()
    aq2["country_norm"] = aq2["country"].apply(_country_iso2)
    aq2["region"] = aq2["region"].astype(str).str.upper()

    # exact merge (country, region)
    df = fac.merge(
        aq2[["country_norm","region","baseline_water_stress"]],
        on=["country_norm","region"], how="left"
    )

    # fallback to country-level ANY
    country_any = (
        aq2[aq2["region"]=="ANY"]
        .set_index("country_norm")["baseline_water_stress"]
        .to_dict()
    )
    if "baseline_water_stress" not in df.columns:
        df["baseline_water_stress"] = pd.NA
    df["baseline_water_stress"] = df.apply(
        lambda r: country_any.get(r["country_norm"], r["baseline_water_stress"]),
        axis=1
    )

    # numeric & flags
    df["weight"] = pd.to_numeric(df.get("weight", 1.0), errors="coerce").fillna(0.0)
    df["baseline_water_stress"] = pd.to_numeric(df["baseline_water_stress"], errors="coerce").fillna(2.5)
    df["hs_flag"] = (df["baseline_water_stress"] >= 4.0).astype(float)

    def agg(x: pd.DataFrame) -> pd.Series:
        w = x["weight"].sum() or 1e-9
        return pd.Series({
            "w_stress_weighted": (x["baseline_water_stress"] * x["weight"]).sum() / w,
            "pct_ops_high_stress": (x["hs_flag"] * x["weight"]).sum() / w
        })

    out = df.groupby("symbol", sort=False)[["baseline_water_stress","hs_flag","weight"]].apply(agg).reset_index()
    return out
