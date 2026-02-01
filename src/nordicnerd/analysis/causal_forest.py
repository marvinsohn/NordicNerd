import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from nordicnerd.config import BLD

# =========================================
# Pfade
# =========================================
DATA_PATH = BLD / "data/race_data_processed.pkl"

# =========================================
# Daten einlesen
# =========================================
df = pd.read_pickle(DATA_PATH)

# =========================================
# Ziel- und Treatment-Variablen
# =========================================
Y = df["misses"].astype(float)
T = df["is_aggressive"].astype(int)

# =========================================
# Features auswählen (nur existierende)
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

# Entferne Zeilen mit NaNs in Treatment, Outcome oder Features
all_needed_cols = ["is_aggressive", "misses"] + candidate_X_cols
df_clean = df.dropna(subset=all_needed_cols)

Y = df_clean["misses"].astype(float)
T = df_clean["is_aggressive"].astype(int)
X = df_clean[candidate_X_cols]

# =========================================
# Preprocessing: numerische bleiben, kategorische One-Hot
# =========================================
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocess_X = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

X_proc = preprocess_X.fit_transform(X)

# =========================================
# Causal Forest Modell aufsetzen
# =========================================
cf = CausalForestDML(
    model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42),
    model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42),
    n_estimators=1000,
    min_samples_leaf=5,
    random_state=42,
    verbose=0
)

# =========================================
# Modell fitten
# =========================================
cf.fit(Y, T, X=X_proc)

# =========================================
# Treatment Effects schätzen
# =========================================
te_pred = cf.effect(X_proc)
print("Estimated treatment effects (first 10):", te_pred[:10])

# =========================================
# Numerische Summary
# =========================================
print("ATE (mean TE):", te_pred.mean())
print("Std of TE:", te_pred.std())
print("5%, 50%, 95% quantiles:", np.quantile(te_pred, [0.05, 0.5, 0.95]))

# =========================================
# Heterogenität: Verteilung der Effekte
# =========================================
plt.hist(te_pred, bins=40)
plt.xlabel("Individual Treatment Effect")
plt.ylabel("Frequency")
plt.title("Distribution of Treatment Effects")
plt.tight_layout()
#plt.show()

# =========================================
# CATE preparation
# =========================================
df_cate = df.loc[X.index].copy()
df_cate["ite"] = te_pred

df_cate["time_behind_bin"] = pd.qcut(
    df_cate["time_behind_before"],
    q=4,
    labels=["low", "mid-low", "mid-high", "high"]
)

cate_time_behind = (
    df_cate
    .groupby("time_behind_bin")["ite"]
    .agg(["mean", "std", "count"])
)

print("\nCATE by time behind leader:")
print(cate_time_behind)

cate_time_behind.to_csv(
    BLD / "results" / "cate_time_behind.csv"
)

df_cate["form_bin"] = pd.qcut(
    df_cate["ski_form_season_z"],
    q=3,
    labels=["low form", "mid form", "high form"]
)

cate_form = (
    df_cate
    .groupby("form_bin")["ite"]
    .agg(["mean", "std", "count"])
)

print("\nCATE by season form:")
print(cate_form)

cate_form.to_csv(
    BLD / "results" / "cate_form.csv"
)

df_cate["rank_bin"] = pd.qcut(
    df_cate["rank_before_shooting"],
    q=4,
    labels=["front pack", "upper mid", "lower mid", "back pack"]
)

cate_rank = (
    df_cate
    .groupby("rank_bin", observed=False)["ite"]
    .agg(["mean", "std", "count"])
)

print("\nCATE by rank before shooting:")
print(cate_rank)

cate_rank.to_csv(BLD / "cate_rank_before_shooting.csv")

cate_shooting_number = (
    df_cate
    .groupby("shooting_number", observed=False)["ite"]
    .agg(["mean", "std", "count"])
    .sort_index()
)

print("\nCATE by shooting number:")
print(cate_shooting_number)

cate_shooting_number.to_csv(BLD / "cate_shooting_number.csv")
