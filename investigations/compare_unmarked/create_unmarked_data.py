import pandas as pd
import numpy as np
import os
from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset_using_env_var,
    random_checklist_subset,
)
import sys

ebird = load_ebird_dataset_using_env_var()
subset_size = int(sys.argv[1])
target_dir = sys.argv[2]

os.makedirs(target_dir, exist_ok=True)

obs_covs_full = ebird["train"].X_obs
cell_ids_full = ebird["train"].env_cell_ids
subset = random_checklist_subset(obs_covs_full.shape[0], cell_ids_full, subset_size)

X_env = ebird["train"].X_env.iloc[subset["env_cell_indices"]]
X_env = X_env[[x for x in X_env.columns if "bio" in x]]
obs_covs = obs_covs_full.iloc[subset["checklist_indices"]]
y_checklist = (
    ebird["train"].y_obs["Cardinalis cardinalis"].iloc[subset["checklist_indices"]]
).values
cell_ids = subset["checklist_cell_ids"]
obs_covs["cell_id"] = cell_ids

X_env.to_csv(os.path.join(target_dir, "X_env_orig.csv"))
obs_covs.to_csv(os.path.join(target_dir, "obs_covs_orig.csv"))
pd.Series(y_checklist).to_csv(os.path.join(target_dir, "y_orig.csv"))
pd.Series(cell_ids).to_csv(os.path.join(target_dir, "cell_ids_orig.csv"))

results = dict()

for sample_name in ["protocol_type", "log_duration", "daytimes_alt"]:

    flat_version = obs_covs.groupby("cell_id").apply(
        lambda x: pd.Series({i: y for i, y in enumerate(x[sample_name].values)})
    )

    flat = flat_version.reset_index()

    wide_version = flat.pivot(index="cell_id", columns="level_1", values=0)

    results[sample_name] = wide_version


for cur_obs_cov_name, cur_obs_df in results.items():

    cur_obs_df.to_csv(target_dir + cur_obs_cov_name + ".csv")

X_env.iloc[wide_version.index].to_csv(target_dir + "env_cov_unmarked.csv")

presence_df = pd.DataFrame(y_checklist)
presence_df["cell_id"] = cell_ids

presences = (
    presence_df.groupby("cell_id")
    .apply(lambda x: pd.Series({i: y.astype(int) for i, y in enumerate(x[0].values)}))
    .reset_index()
)

wide_presences = presences.pivot(index="cell_id", columns="level_1", values=0)

wide_presences.to_csv(target_dir + "sample_observations.csv")
