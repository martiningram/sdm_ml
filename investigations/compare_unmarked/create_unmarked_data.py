import pandas as pd
import numpy as np
import os
from utils import fetch_data


obs_covs, arrays = fetch_data()

results = dict()

for sample_name in ["protocol_type", "log_duration", "time_of_day"]:

    flat_version = obs_covs.groupby("cell_id").apply(
        lambda x: pd.Series({i: y for i, y in enumerate(x[sample_name].values)})
    )

    flat = flat_version.reset_index()

    wide_version = flat.pivot(index="cell_id", columns="level_1", values=0)

    results[sample_name] = wide_version

target_dir = "./unmarked_data/"

os.makedirs(target_dir, exist_ok=True)

for cur_obs_cov_name, cur_obs_df in results.items():

    cur_obs_df.to_csv(target_dir + cur_obs_cov_name + ".csv")

arrays["env_covariates"].iloc[wide_version.index].to_csv(
    target_dir + "env_cov_unmarked.csv"
)

presence_df = pd.DataFrame(arrays["is_present"])
presence_df["cell_id"] = arrays["numeric_checklist_cell_ids"]

presences = (
    presence_df.groupby("cell_id")
    .apply(lambda x: pd.Series({i: y.astype(int) for i, y in enumerate(x[0].values)}))
    .reset_index()
)

wide_presences = presences.pivot(index="cell_id", columns="level_1", values=0)

wide_presences.to_csv(target_dir + "sample_observations.csv")
