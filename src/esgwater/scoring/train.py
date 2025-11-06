from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import joblib

PROCESSED = Path("data/processed")
RAW = Path("data/raw")


def _ensure_sector(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # if sector already present and not all NaN, keep
    if "sector" in out.columns and out["sector"].notna().any():
        out["sector"] = out["sector"].fillna("UNK").astype(str)
        return out

    # try to load from manual_features.csv
    mf_p = RAW / "manual_features.csv"
    if mf_p.exists():
        hdr = pd.read_csv(mf_p, nrows=0).columns.tolist()
        use = [c for c in ["symbol", "sector"] if c in hdr]
        if "sector" in use:
            mf = pd.read_csv(mf_p, usecols=use)
            out = out.drop(columns=[c for c in ["sector"] if c in out.columns], errors="ignore")
            out = out.merge(mf[["symbol", "sector"]], on="symbol", how="left")

    # fallback
    if "sector" not in out.columns:
        out["sector"] = "UNK"

    out["sector"] = out["sector"].fillna("UNK").astype(str)
    return out


def _within_group_z(s: pd.Series, g: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    g = g.astype(str)
    out = pd.Series(index=x.index, dtype=float)
    for k, sub in x.groupby(g):
        mu = sub.mean()
        sd = sub.std()
        out.loc[sub.index] = 0.0 if (sd is None or sd == 0 or np.isnan(sd)) else (sub - mu) / sd
    return out


def build_model_features() -> pd.DataFrame:
    eng = pd.read_parquet(PROCESSED / "engineered_water_features.parquet")
    univ = pd.read_parquet(PROCESSED / "universe_filtered.parquet")
    lab = pd.read_csv(RAW / "labels.csv")

    # merge (bring sector if present in universe)
    cols = ["symbol"] + ([c for c in ["sector"] if "sector" in univ.columns])
    base = eng.merge(univ[cols], on="symbol", how="left")
    df = base.merge(lab, on="symbol", how="inner")
    df = _ensure_sector(df)

    # numeric coercions
    for c in ["w_stress_weighted", "pct_ops_high_stress", "water_vulnerability"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # fundamentals (optional)
    fund_p = RAW / "fundamentals.csv"
    if fund_p.exists():
        fd = pd.read_csv(fund_p)
        for c in ["pb", "roe", "profit_margin"]:
            if c in fd.columns:
                fd[c] = pd.to_numeric(fd[c], errors="coerce")
        df = df.merge(fd, on="symbol", how="left")

    # engineered extras
    df["bm"] = np.where((df.get("pb") > 0) & (~pd.isna(df.get("pb"))), -np.log(df["pb"]), np.nan)
    df["stress_x_vuln"] = df["w_stress_weighted"] * df["water_vulnerability"]
    df["stress_sq"] = np.power(df["w_stress_weighted"], 2)
    df["hs_flag"] = (df["pct_ops_high_stress"] > 0).astype(float)

    # within-sector z-scores
    sec = df["sector"].fillna("UNK").astype(str)
    for c in ["w_stress_weighted", "pct_ops_high_stress", "water_vulnerability", "bm", "roe", "profit_margin"]:
        if c in df.columns:
            df[f"{c}_z_sec"] = _within_group_z(df[c], sec)

    # feature set (prefer z-scored where available)
    use = [c for c in [
        "w_stress_weighted_z_sec",
        "pct_ops_high_stress_z_sec",
        "water_vulnerability_z_sec",
        "bm_z_sec",
        "roe_z_sec",
        "profit_margin_z_sec",
        "stress_x_vuln",
        "stress_sq",
        "hs_flag",
    ] if c in df.columns]

    X = df[["symbol", "sector"] + use].copy()

    # sector-neutral rank target (dense rank within sector → 0..1)
    tgt = pd.to_numeric(df["target"], errors="coerce")
    ranks = []
    for k, sub in df.assign(_tgt=tgt).groupby(sec):
        r = sub["_tgt"].rank(pct=True, method="dense")
        ranks.append(pd.Series(r.values, index=sub.index))
    X["target_rank"] = pd.concat(ranks).sort_index().values

    PROCESSED.mkdir(parents=True, exist_ok=True)
    X.to_parquet(PROCESSED / "model_features.parquet", index=False)
    return X


def _cv_eval(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> dict:
    n_groups = int(groups.nunique())
    n_splits = max(2, min(5, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    r2s, ics, ics_sec = [], [], []

    for tr, te in gkf.split(X, y, groups):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])

        r2s.append(r2_score(y.iloc[te], pred))
        ics.append(spearmanr(y.iloc[te], pred).statistic)

        sec_te = groups.iloc[te].astype(str).values
        df_te = pd.DataFrame({"y": y.iloc[te].values, "p": pred, "sec": sec_te})
        res = []
        for _, sub in df_te.groupby("sec"):
            if len(sub) < 2:
                continue
            ic_k = spearmanr(sub["y"], sub["p"]).statistic
            if not np.isnan(ic_k):
                res.append(ic_k)
        ics_sec.append(np.mean(res) if res else np.nan)

    return {
        "r2_cv_mean": float(np.nanmean(r2s)),
        "r2_cv_std": float(np.nanstd(r2s)),
        "ic_cv_mean": float(np.nanmean(ics)),
        "ic_cv_std": float(np.nanstd(ics)),
        "ic_sec_mean": float(np.nanmean(ics_sec)),
        "ic_sec_std": float(np.nanstd(ics_sec)),
        "n_splits": n_splits,
    }


def train_and_save():
    mf = build_model_features()

    # groups = sector (robust)
    groups = mf["sector"].fillna("UNK").astype(str)

    # numeric block
    num = mf.select_dtypes("number").fillna(0.0)
    if "target_rank" not in num.columns:
        raise SystemExit("No target_rank column in model_features.")
    y = num.pop("target_rank")

    # ensure at least 2 obs per group; otherwise skip CV
    vc = groups.value_counts()
    keep = groups.isin(vc[vc >= 2].index)

    if keep.sum() >= 6 and groups[keep].nunique() >= 2:
        Xc, yc, gc = num[keep].reset_index(drop=True), y[keep].reset_index(drop=True), groups[keep].reset_index(drop=True)
        do_cv = True
    else:
        Xc, yc, gc = num, y, groups
        do_cv = False

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("gbr", GradientBoostingRegressor(
            n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=7
        )),
    ])

    meta_cv = {"cv": "skipped"}
    if do_cv and len(Xc) >= 4:
        try:
            meta_cv = _cv_eval(pipe, Xc, yc, gc)
            meta_cv["cv"] = f"GroupKFold by sector ({meta_cv['n_splits']})"
        except Exception:
            meta_cv = {"cv": "skipped"}

    # fit final model on ALL rows
    pipe.fit(num, y)
    joblib.dump(pipe, PROCESSED / "model.joblib")

    meta = {
        **meta_cv,
        "n_samples": int(len(mf)),
        "n_features": int(num.shape[1]),
        "features": list(num.columns),
    }
    with open(PROCESSED / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("✅ Saved model.joblib & train_meta.json", meta)


if __name__ == "__main__":
    train_and_save()
