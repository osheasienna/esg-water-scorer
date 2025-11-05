
import pandas as pd
from esgwater.features.water_ops import load_facilities, load_aqueduct_lookup, facility_water_stress, load_sector_vuln, attach_sector_vulnerability

fac = load_facilities()
aq  = load_aqueduct_lookup()
sec = load_sector_vuln()

stress = facility_water_stress(fac, aq)
print("STRESS rows:", len(stress), "symbols:", stress["symbol"].nunique())
print(stress.describe(include="all"))

try:
    eng = pd.read_parquet("data/processed/engineered_water_features.parquet")
    print("\nENGINEERED rows:", len(eng))
    print(eng[["w_stress_weighted","pct_ops_high_stress"]].describe())
except Exception as e:
    print("No engineered parquet yet:", e)
