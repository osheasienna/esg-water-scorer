import pandas as pd
import yaml
from pathlib import Path


def load_config():
    indicators = yaml.safe_load(open("config/indicators.yml"))
    weights = yaml.safe_load(open("config/weights.yml"))
    return indicators, weights


def load_universe():
    return pd.read_parquet("data/processed/universe_latest.parquet")


def load_manual_features():
    path = Path("data/raw/manual_features.csv")
    if not path.exists():
        return pd.DataFrame(columns=["symbol"])
    return pd.read_csv(path)


def normalize(col, higher_is_better=True):
    if col.isna().all():
        return pd.Series([None] * len(col), index=col.index)

    x = col.astype(float).copy()
    if not higher_is_better:
        x = -x

    lo, hi = x.min(), x.max()
    if lo == hi:
        return pd.Series([0.5] * len(col), index=col.index)

    return (x - lo) / (hi - lo)


def score():
    indicators_cfg, weights_cfg = load_config()
    indicators = indicators_cfg["indicators"]

    univ = load_universe()
    # ensure symbol column exists
    if "symbol" not in univ.columns and "ticker" in univ.columns:
        univ = univ.rename(columns={"ticker": "symbol"})

    feats = load_manual_features()

    df = univ.merge(feats, how="left", on="symbol")

    # indicator scores in [0,1]
    for name, cfg in indicators.items():
        col = cfg["column"]
        higher = cfg.get("higher_is_better", True)
        if col not in df.columns:
            df[name] = pd.NA
        else:
            df[name] = normalize(df[col], higher)

    pweights = weights_cfg["pillar_weights"]
    iweights = weights_cfg["indicator_weights"]

    df["water_score"] = (
        (df["water_intensity"] * iweights.get("water_intensity", 0)).fillna(0)
        + (df["basin_stress"] * iweights.get("basin_stress", 0)).fillna(0)
        + (df["water_targets"] * iweights.get("water_targets", 0)).fillna(0)
    )

    df["env_score_pillar"] = (df["env_score"] * iweights.get("env_score", 0)).fillna(0)
    df["social_score_pillar"] = (df["social_score"] * iweights.get("social_score", 0)).fillna(0)
    df["gov_score_pillar"] = (df["gov_score"] * iweights.get("gov_score", 0)).fillna(0)

    df["overall_score"] = (
        df["water_score"] * pweights.get("water", 0)
        + df["env_score_pillar"] * pweights.get("env", 0)
        + df["social_score_pillar"] * pweights.get("social", 0)
        + df["gov_score_pillar"] * pweights.get("gov", 0)
    )

    df["rank"] = df["overall_score"].rank(ascending=False, method="dense")
    return df.sort_values(["rank", "symbol"])
