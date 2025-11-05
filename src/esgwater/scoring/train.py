
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
import joblib

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

def _ensure_sector(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sector" not in out.columns:
        out["sector"] = "UNK"
    out["sector"] = out["sector"].fillna("UNK").astype(str)
    return out

def _build_group_key(mf: pd.DataFrame) -> pd.Series:
    exch = mf["symbol"].astype(str).str.split(".").str[-1].fillna("NA")  # proxy exchange suffix
    return mf["sector"].astype(str) + "|" + exch

def build_model_features() -> pd.DataFrame:
    eng = pd.read_parquet(PROCESSED / "engineered_water_features.parquet")
    eng = _ensure_sector(eng)
    # filter to your pre-built universe
    filt = pd.read_parquet(PROCESSED / "universe_filtered.parquet")
    eng = eng[eng["symbol"].isin(filt["symbol"].astype(str))].reset_index(drop=True)

    lab = pd.read_csv(RAW / "labels.csv")
    df = eng.merge(lab, on="symbol", how="inner")

    # base features
    df["w_stress_weighted"]   = pd.to_numeric(df["w_stress_weighted"], errors="coerce")
    df["pct_ops_high_stress"] = pd.to_numeric(df["pct_ops_high_stress"], errors="coerce")
    df["water_vulnerability"] = pd.to_numeric(df["water_vulnerability"], errors="coerce")

    # interactions / nonlinearity
    df["stress_x_vuln"] = df["w_stress_weighted"] * df["water_vulnerability"]
    df["stress_sq"]     = df["w_stress_weighted"] ** 2
    df["hs_flag"]       = (df["pct_ops_high_stress"] > 0).astype(float)

    num_cols = [
        "w_stress_weighted","pct_ops_high_stress","water_vulnerability",
        "stress_x_vuln","stress_sq","hs_flag"
    ]
    X = df[["symbol","sector"] + num_cols].copy()
    X["target"] = pd.to_numeric(df["target"], errors="coerce")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    X.to_parquet(PROCESSED / "model_features.parquet", index=False)
    return X

def train_and_save():
    mf = build_model_features()

    # groups: sector|exchange
    groups = _build_group_key(mf).fillna("UNK")

    # drop rows with missing numerics
    num = mf.select_dtypes("number").fillna(0.0)
    y = num.pop("target")

    # require at least 2 obs per group
    ok_groups = groups.value_counts()
    keep = groups.isin(ok_groups[ok_groups >= 2].index)
    mf  = mf[keep].reset_index(drop=True)
    num = num[keep].reset_index(drop=True)
    y   = y[keep].reset_index(drop=True)
    groups = groups[keep].reset_index(drop=True)

    # dynamic n_splits
    n_groups = int(groups.nunique())
    n_splits = max(2, min(5, n_groups))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, random_state=7
        ))
    ])

    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(pipe, num, y, cv=cv.split(num, y, groups), scoring="r2")

    pipe.fit(num, y)
    joblib.dump(pipe, PROCESSED / "model.joblib")

    meta = {
        "cv": f"GroupKFold by sector|exch ({n_splits})",
        "r2_cv_mean": float(scores.mean()) if len(scores)>0 else None,
        "r2_cv_std": float(scores.std()) if len(scores)>0 else None,
        "n_samples": int(len(mf)),
        "n_features": int(num.shape[1]),
        "features": list(num.columns),
        "n_groups": n_groups
    }
    with open(PROCESSED / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved model.joblib & train_meta.json", meta)

if __name__ == "__main__":
    train_and_save()
