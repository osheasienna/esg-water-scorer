
from __future__ import annotations
import joblib, shap, pandas as pd
from pathlib import Path
from esgwater.models.train import load_training_table, FEATURES_NUM, FEATURES_CAT

def explain(sample_symbols=None, model_path="models/esgwater/model.joblib"):
    pipe = joblib.load(model_path)
    df = load_training_table()
    if sample_symbols is not None:
        df = df[df["symbol"].isin(sample_symbols)].copy()
    X = df[[c for c in set(FEATURES_NUM+FEATURES_CAT) if c in df.columns]]
    # preprocess using pipeline's preprocessor to align columns
    pre = pipe.named_steps["pre"]
    X_pre = pre.fit_transform(X)
    model = pipe.named_steps["rf"]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_pre)
    imp = getattr(sv, "abs", lambda: abs(sv))().mean(axis=0)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature_index": range(len(imp)), "abs_mean_shap": imp}).to_csv(
        "data/processed/shap_global.csv", index=False
    )
    return "data/processed/shap_global.csv"
