
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

FEATURES_NUM = [
    "water_intensity","baseline_water_stress","water_targets",
    "env_score","social_score","gov_score",
    "w_stress_weighted","pct_ops_high_stress","water_vulnerability"
]
FEATURES_CAT = ["sector"]

def _normalize_inverse(x: pd.Series) -> pd.Series:
    # Higher is better after inversion; if constant, return 0.5
    x = x.astype(float)
    xmin, xmax = float(x.min()), float(x.max())
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return pd.Series(0.5, index=x.index)
    if xmax - xmin <= 1e-12:
        return pd.Series(0.5, index=x.index)
    z = (x - xmin) / (xmax - xmin)
    return 1.0 - z

def _water_proxy_target(df: pd.DataFrame) -> pd.Series:
    # Higher is better
    wstress = df.get("w_stress_weighted", pd.Series(2.5, index=df.index)).fillna(2.5)
    phigh   = df.get("pct_ops_high_stress", pd.Series(0.0, index=df.index)).fillna(0.0)
    wint    = df.get("water_intensity", pd.Series(np.nan, index=df.index))
    # fill intensity with median if missing; if all missing, set to 0
    med = float(wint.median()) if np.isfinite(wint.median()) else 0.0
    wint = wint.fillna(med)
    return (
        (1.0 - (wstress/5.0).clip(0,1)) * 0.50 +
        (1.0 - phigh.clip(0,1))        * 0.30 +
        _normalize_inverse(wint).clip(0,1) * 0.20
    )

def load_training_table() -> pd.DataFrame:
    df = pd.read_parquet("data/processed/universe_latest.parquet")
    mf = Path("data/raw/manual_features.csv")
    if mf.exists():
        df = df.merge(pd.read_csv(mf), on="symbol", how="left")
    eng = Path("data/processed/engineered_water_features.parquet")
    if eng.exists():
        df = df.merge(pd.read_parquet(eng), on="symbol", how="left")
    # primary target: average ESG pillars
    df["y_target"] = df[["env_score","social_score","gov_score"]].mean(axis=1)
    # if too sparse, use water-proxy for ALL rows (consistent target)
    if df["y_target"].notna().sum() < 50:
        df["y_target"] = _water_proxy_target(df)
    return df

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_ix = [c for c in FEATURES_NUM if c in X.columns]
    cat_ix = [c for c in FEATURES_CAT if c in X.columns]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_ix),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_ix),
    ], remainder="drop")
    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    return Pipeline([("pre", pre), ("rf", rf)])

def train(save_dir="models/esgwater"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df = load_training_table()
    X = df[[c for c in set(FEATURES_NUM+FEATURES_CAT) if c in df.columns]].copy()
    y = df["y_target"].astype(float).copy()

    # holdout; if dataset tiny, still proceed
    test_size = 0.25 if len(df) >= 8 else 0.34 if len(df) >= 6 else 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

    pipe = build_pipeline(X)
    pipe.fit(Xtr, ytr)
    joblib.dump(pipe, f"{save_dir}/model.joblib")

    # compute R2 only if test has >=2 samples
    r2 = None
    if len(yte) >= 2:
        try:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(yte, pipe.predict(Xte)))
        except Exception:
            r2 = None

    metrics = {"r2": r2, "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    # JSON-safe
    for k, v in list(metrics.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            metrics[k] = None
    Path(f"{save_dir}/metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

def predict(save_path="data/processed/ml_scores.parquet", model_path="models/esgwater/model.joblib"):
    pipe: Pipeline = joblib.load(model_path)  # type: ignore
    df = load_training_table()
    X = df[[c for c in set(FEATURES_NUM+FEATURES_CAT) if c in df.columns]].copy()
    preds = pipe.predict(X)
    out = df[["symbol","name"]].copy() if "name" in df.columns else df[["symbol"]].copy()
    out["ml_score_raw"] = preds
    lo, hi = float(out["ml_score_raw"].min()), float(out["ml_score_raw"].max())
    out["ml_score"] = 50.0 if hi == lo else (out["ml_score_raw"] - lo) / (hi - lo) * 100.0
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out.to_parquet(save_path, index=False)
    return save_path
