# ==================================================
# Causal ML: Effect of Aggressive Shooting on Miss Rate
# ==================================================

import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from econml.dml import CausalForestDML

# --------------------------------------------------
# Load processed data
# --------------------------------------------------

df = pd.read_pickle(
    "BLD/data/race_data_processed.pkl"
)

# --------------------------------------------------
# Outcome: miss rate
# --------------------------------------------------

df["miss_rate"] = df["misses"] / df["shots"]

# Drop impossible rows
df = df[(df["shots"] > 0) & (df["miss_rate"].notna())]

# --------------------------------------------------
# Treatment
# --------------------------------------------------

T = df["is_aggressive_quantile"].astype(int)

# --------------------------------------------------
# Covariates (pre-treatment only)
# --------------------------------------------------

X = df[
    [
        "rank_before_shooting",
        "time_behind_before_sec",
        "ski_form_season_z",
        "ski_form_race_z",
        "shooting_position",
        "shooting_number",
    ]
]

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_features = [
    "rank_before_shooting",
    "time_behind_before_sec",
    "ski_form_season_z",
    "ski_form_race_z",
    "shooting_number",
]

categorical_features = ["shooting_position"]

preprocess_X = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False),
            categorical_features,
        ),
    ]
)

# --------------------------------------------------
# Models for DML
# --------------------------------------------------

model_y = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
)

model_t = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
)

# --------------------------------------------------
# Causal Forest
# --------------------------------------------------

cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    n_estimators=800,
    min_samples_leaf=15,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

# --------------------------------------------------
# Fit
# --------------------------------------------------

X_proc = preprocess_X.fit_transform(X)

cf.fit(
    Y=df["miss_rate"].values,
    T=T.values,
    X=X_proc,
)

# --------------------------------------------------
# Average Treatment Effect
# --------------------------------------------------

ate = cf.ate(X_proc)
ate_lb, ate_ub = cf.ate_interval(X_proc)

print("ATE (aggressive shooting → miss rate):")
print(f"  {ate:.4f}")
print(f"95% CI: [{ate_lb:.4f}, {ate_ub:.4f}]")

# --------------------------------------------------
# Individual Treatment Effects
# --------------------------------------------------

df["ite_miss_rate"] = cf.effect(X_proc)

print("\nITE summary:")
print(df["ite_miss_rate"].describe())

# --------------------------------------------------
# Save model + results
# --------------------------------------------------

with open("BLD/models/causal_forest_miss_rate.pkl", "wb") as f:
    pickle.dump(
        {
            "model": cf,
            "preprocess_X": preprocess_X,
            "ate": ate,
            "ate_ci": (ate_lb, ate_ub),
        },
        f,
    )

df[["ite_miss_rate"]].to_pickle(
    "BLD/data/ite_miss_rate.pkl"
)
