from sklearn.preprocessing import StandardScaler
from sdm_ml.checklist_level.functional.max_lik_occu_model import fit
from ml_tools.patsy import create_formula
import pandas as pd
import numpy as np
import time
import sys
from os.path import join
import os


data_dir = sys.argv[1]
target_dir = sys.argv[2]

X_env = pd.read_csv(join(data_dir, "X_env_orig.csv"), index_col=0)
X_obs = pd.read_csv(join(data_dir, "obs_covs_orig.csv"), index_col=0)
y = pd.read_csv(join(data_dir, "y_orig.csv"), index_col=0)["0"].values.astype(int)
cell_ids = pd.read_csv(join(data_dir, "cell_ids_orig.csv"), index_col=0)[
    "0"
].values.astype(int)

scaler = StandardScaler()

X_env_scaled = pd.DataFrame(scaler.fit_transform(X_env), columns=X_env.columns)

env_formula = create_formula(
    X_env_scaled.columns, main_effects=True, quadratic_effects=True, interactions=False
)
obs_formula = "protocol_type + daytimes_alt + log_duration"

start_time = time.time()

results = fit(
    X_env_scaled,
    X_obs,
    y,
    cell_ids,
    env_formula,
    obs_formula,
)

time_taken = time.time() - start_time

os.makedirs(target_dir, exist_ok=True)

results["env_coefs"].to_csv(join(target_dir, "fast_occu_env_coefs.csv"))
results["obs_coefs"].to_csv(join(target_dir, "fast_occu_obs_coefs.csv"))

print(time_taken, file=open(join(target_dir, "fast_occu_runtime.txt"), "w"))
