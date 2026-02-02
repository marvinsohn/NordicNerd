import pandas as pd
import numpy as np
from pathlib import Path
from econml.dml import CausalForestDML
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


def run_causal_forest_analysis(data_path: Path, output_dir: Path):
    """
    Runs causal forest estimation and CATE analysis.
    Saves results to output_dir and returns summary statistics.
    """

    # =========================================
    # Load data
    # =========================================
    df = pd.read_pickle(data_path)

    # =========================================
    # Outcome & treatment
    # =========================================
    candidate_X_cols = [
        "air_temp",
        "snow_temp",
        "ski_form_race_z",
        "ski_form_season_z",
        "rank_before_shooting",
        "shooting_number",
        "shooting_position",
        "time_behind_before"
    ]

    all_needed_cols = ["is_aggressive", "misses"] + candidate_X_cols
    df_clean = df.dropna(subset=all_needed_cols)

    Y = df_clean["misses"].astype(float)
    T = df_clean["is_aggressive"].astype(int)
    X = df_clean[candidate_X_cols]

    # =========================================
    # Preprocessing
    # =========================================
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocess_X = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    X_proc = preprocess_X.fit_transform(X)

    # =========================================
    # Causal Forest
    # =========================================
    cf = CausalForestDML(
        model_t=RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42
        ),
        model_y=RandomForestRegressor(
            n_estimators=100, min_samples_leaf=10, random_state=42
        ),
        n_estimators=1000,
        min_samples_leaf=5,
        random_state=42,
        verbose=0,
    )

    cf.fit(Y, T, X=X_proc)

    te_pred = cf.effect(X_proc)

    # =========================================
    # ATE summary
    # =========================================
    ate_summary = {
        "ate_mean": float(te_pred.mean()),
        "ate_std": float(te_pred.std()),
        "q05": float(np.quantile(te_pred, 0.05)),
        "q50": float(np.quantile(te_pred, 0.50)),
        "q95": float(np.quantile(te_pred, 0.95)),
    }

    pd.DataFrame([ate_summary]).to_csv(
        output_dir / "ate_summary.csv", index=False
    )

    # =========================================
    # CATE preparation
    # =========================================
    df_cate = df_clean.copy()
    df_cate["ite"] = te_pred

    # ---- time behind leader
    df_cate["time_behind_bin"] = pd.qcut(
        df_cate["time_behind_before"],
        q=4,
        labels=["low", "mid-low", "mid-high", "high"],
    )

    cate_time = (
        df_cate.groupby("time_behind_bin", observed=False)["ite"]
        .agg(["mean", "std", "count"])
    )
    cate_time.to_csv(output_dir / "cate_time_behind.csv")

    # ---- season form
    df_cate["form_bin"] = pd.qcut(
        df_cate["ski_form_season_z"],
        q=3,
        labels=["low form", "mid form", "high form"],
    )

    cate_form = (
        df_cate.groupby("form_bin", observed=False)["ite"]
        .agg(["mean", "std", "count"])
    )
    cate_form.to_csv(output_dir / "cate_form.csv")

    # ---- rank before shooting
    df_cate["rank_bin"] = pd.qcut(
        df_cate["rank_before_shooting"],
        q=4,
        labels=["front pack", "upper mid", "lower mid", "back pack"],
    )

    cate_rank = (
        df_cate.groupby("rank_bin", observed=False)["ite"]
        .agg(["mean", "std", "count"])
    )
    cate_rank.to_csv(output_dir / "cate_rank_before_shooting.csv")

    # ---- shooting number
    cate_shooting = (
        df_cate.groupby("shooting_number", observed=False)["ite"]
        .agg(["mean", "std", "count"])
        .sort_index()
    )
    cate_shooting.to_csv(output_dir / "cate_shooting_number.csv")

    return ate_summary
