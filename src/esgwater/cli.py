from esgwater.scoring.train import train_and_save
from pathlib import Path
import click

@click.group()
def cli():
    """ESGWater CLI"""
    pass

@cli.command()
def features():
    """Build engineered features (sector + regional water)."""
    from esgwater.features.build_features import build_engineered_features
    path = build_engineered_features()
    print(f"✅ Built features → {path}")

@cli.command()
def train():
    train_and_save()

@cli.command()
def predict():
    """Predict ML scores and write to parquet."""
    from esgwater.models.train import predict as _predict
    path = _predict()
    print(f"✅ Predicted → {path}")

@cli.command()
def rank():
    """Write ML-ranked CSV sorted by ml_score (desc)."""
    import pandas as pd
    out = "data/processed/rankings_ml.csv"
    ml = pd.read_parquet("data/processed/ml_scores.parquet").dropna(subset=["ml_score"])
    ml = ml.sort_values("ml_score", ascending=False)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ml.to_csv(out, index=False)
    print(f"✅ Wrote {out}")

@cli.command()
@click.option("--symbols", default="", help="Comma-separated tickers to explain")
def explain(symbols: str):
    """Write SHAP global importance (and optionally for sample)."""
    from esgwater.models.explain import explain as _explain
    syms = [s.strip() for s in symbols.split(",")] if symbols else None
    path = _explain(sample_symbols=syms)
    print(f"✅ Wrote SHAP summary → {path}")

@cli.command()
def backtest():
    """Very simple placeholder backtest based on ML score quintiles."""
    import pandas as pd, numpy as np
    ml = pd.read_parquet("data/processed/ml_scores.parquet")
    np.random.seed(42)
    ml = ml.dropna(subset=["ml_score"]).copy()
    ml["ret_next"] = np.random.normal(0.001, 0.02, len(ml))
    q = pd.qcut(ml["ml_score"], 5, labels=False, duplicates="drop")
    out = (ml.assign(quintile=q).groupby("quintile")["ret_next"].agg(["mean","std","count"]).reset_index())
    out.to_csv("data/processed/backtest_quintiles.csv", index=False)
    print("✅ Backtest → data/processed/backtest_quintiles.csv")

if __name__ == "__main__":
    cli()
