import numpy as np
import pandas as pd
from sdm_ml.checklist_level.functional.max_lik_occu_model import fit

df = pd.read_csv("./data.csv", index_col=0)
env_covs = df[["elev", "forest", "length"]]

dates = df[["date.1", "date.2", "date.3"]].values.reshape(-1)
ivels = df[["ivel.1", "ivel.2", "ivel.3"]].values.reshape(-1)
ys = df[["y.1", "y.2", "y.3"]].values.reshape(-1)
cell_ids = np.tile(np.arange(df.shape[0]), (3, 1)).T.reshape(-1)

obs_data = np.stack([dates, ivels, cell_ids, ys], axis=1)

obs_df = pd.DataFrame(obs_data, columns=["date", "ivel", "cell_id", "y"]).dropna()

obs_covs = obs_df[["date", "ivel"]]
cell_ids = obs_df["cell_id"].values.astype(int)
present = obs_df["y"].values.astype(int)

results = fit(
    X_env=env_covs,
    X_checklist=obs_covs,
    y=present,
    cell_ids=cell_ids,
    env_formula="forest + elev + length",
    checklist_formula="date + ivel",
)

results["env_coefs"].to_csv("state_coefs_fast.csv")
results["det_coefs"].to_csv("det_coefs_fast.csv")
