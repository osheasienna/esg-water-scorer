import pandas as pd
from pathlib import Path

fac_p = Path("data/raw/facilities.csv")
fac = pd.read_csv(fac_p) if fac_p.exists() else pd.DataFrame(columns=["country","region"])
countries = sorted(set(fac["country"].dropna().astype(str))) if not fac.empty else []

seed = {
    "US":2.0,"MX":4.0,"CN":3.8,"VN":3.5,"TW":4.6,"IN":4.5,"GB":1.8,"UK":1.8,
    "FR":1.7,"DE":1.6,"ES":3.0,"BR":3.2,"ZA":3.5,"SA":5.0,"AE":5.0,"CL":3.8,
    "AU":2.6,"CA":1.6,"JP":2.4,"KR":3.1,"IT":2.3,"NL":2.1,"SE":1.4,"NO":1.2
}
rows = [{"country": c, "region": "ANY", "baseline_water_stress": seed.get(c, 2.5)}
        for c in (countries or list(seed.keys()))]

Path("data/reference").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("data/reference/aqueduct_lookup.csv", index=False)
print("✅ wrote data/reference/aqueduct_lookup.csv")