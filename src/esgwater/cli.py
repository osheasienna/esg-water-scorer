from pathlib import Path
import click
from esgwater.scoring.train import train_and_save

@click.group()
def cli():
    """ESGWater CLI"""
    pass

@cli.command()
def features():
    """Build engineered water features (sector + regional stress)."""
    from esgwater.features.build_features import build_engineered_features
    out = build_engineered_features()
    print(f"✅ Built features → {out}")

@cli.command()
def train():
    """Train model (builds model_features, runs CV, saves model & meta)."""
    train_and_save()

@cli.command()
def predict():
    """
    Predict ML scores using the SAME feature matrix used in training.
    Writes:
      - data/processed/ml_scores.parquet
      - data/processed/rankings_ml.csv
    """
    import pandas as pd, numpy as np, joblib
    p_proc = Path("data/processed")
    mf_p   = p_proc / "model_features.parquet"
    mdl_p  = p_proc / "model.joblib"
    if not mf_p.exists(): raise SystemExit(f"Missing {mf_p} (run `train` first).")
    if not mdl_p.exists(): raise SystemExit(f"Missing {mdl_p} (run `train` first).")

    mf  = pd.read_parquet(mf_p)
    num = mf.select_dtypes("number").drop(columns=["target_rank"], errors="ignore").fillna(0.0)
    pipe = joblib.load(mdl_p)
    yhat = pipe.predict(num)

    out = pd.DataFrame({"symbol": mf["symbol"], "ml_score_raw": yhat})
    out.to_parquet(p_proc / "ml_scores.parquet", index=False)

    mn, mx = out["ml_score_raw"].min(), out["ml_score_raw"].max()
    if mx > mn:
        out["ml_score"] = 100.0 * (out["ml_score_raw"] - mn) / (mx - mn)
    else:
        out["ml_score"] = (out["ml_score_raw"].rank(pct=True, method="dense")*100)

    try:
        uni = pd.read_parquet(p_proc / "universe_filtered.parquet")[["symbol","name"]]
        out = out.merge(uni, on="symbol", how="left")
    except Exception:
        pass

    out.sort_values("ml_score", ascending=False).to_csv(p_proc / "rankings_ml.csv", index=False)
    print("✅ Predicted → data/processed/ml_scores.parquet")
    print("✅ Wrote → data/processed/rankings_ml.csv")

@cli.command()
def rank():
    """
    Post-process rankings for presentation.
    Writes:
      - data/processed/rankings_ml_percentile.csv
      - data/processed/leaderboard_water_enriched.csv
    """
    import pandas as pd
    p_proc   = Path("data/processed")
    p_rank   = p_proc / "rankings_ml.csv"
    p_feats  = p_proc / "engineered_water_features.parquet"
    p_rank_p = p_proc / "rankings_ml_percentile.csv"
    p_lead   = p_proc / "leaderboard_water_enriched.csv"
    if not p_rank.exists(): raise SystemExit(f"Missing {p_rank} (run `predict` first).")

    r = pd.read_csv(p_rank)
    r["ml_score_pct"] = (r["ml_score_raw"].rank(pct=True, method="average")*100).round(2)
    r.sort_values("ml_score_pct", ascending=False).to_csv(p_rank_p, index=False)

    if p_feats.exists():
        eng = pd.read_parquet(p_feats)
        cols = ["symbol","sector","w_stress_weighted","pct_ops_high_stress","water_vulnerability"]
        r.merge(eng[[c for c in cols if c in eng.columns]], on="symbol", how="left") \
         .sort_values("ml_score_pct", ascending=False).to_csv(p_lead, index=False)
    else:
        r.sort_values("ml_score_pct", ascending=False).to_csv(p_lead, index=False)

    print("✅ Wrote →", p_rank_p)
    print("✅ Wrote →", p_lead)

if __name__ == "__main__":
    cli()
