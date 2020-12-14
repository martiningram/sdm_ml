from sklearn.preprocessing import StandardScaler
from sdm_ml.checklist_level.functional.max_lik_occu_model import fit
from ml_tools.patsy import create_formula
from utils import fetch_data
import pandas as pd
import numpy as np


obs_covs, arrays = fetch_data()

X_env = arrays["env_covariates"]
X_obs = obs_covs

scaler = StandardScaler()

X_env_scaled = pd.DataFrame(scaler.fit_transform(X_env), columns=X_env.columns)

env_formula = create_formula(
    X_env_scaled.columns, main_effects=True, quadratic_effects=True, interactions=False
)
obs_formula = "protocol_type + protocol_type:log_duration + time_of_day + log_duration"

results = fit(
    X_env_scaled,
    X_obs,
    arrays["is_present"],
    arrays["numeric_checklist_cell_ids"],
    env_formula,
    obs_formula,
)

results[0].to_csv("./fast_occu_env_coefs.csv")
results[1].to_csv("./fast_occu_obs_coefs.csv")
