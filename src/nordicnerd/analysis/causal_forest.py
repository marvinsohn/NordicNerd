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
te = cf.effect(X_proc)
print("Estimated treatment effects (first 10):", te[:10])

# =========================================
# Numerische Summary
# =========================================
print("ATE (mean TE):", te.mean())
print("Std of TE:", te.std())
print("5%, 50%, 95% quantiles:", np.quantile(te, [0.05, 0.5, 0.95]))

# =========================================
# Heterogenität: Verteilung der Effekte
# =========================================
plt.hist(te, bins=40)
plt.xlabel("Individual Treatment Effect")
plt.ylabel("Frequency")
plt.title("Distribution of Treatment Effects")
plt.tight_layout()
plt.show()