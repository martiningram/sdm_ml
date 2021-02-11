import sys


model_name = sys.argv[1]
use_double_precision = sys.argv[2] == "double"
use_gpu = sys.argv[3] == "gpu"
subset_size = int(sys.argv[4])
min_presences = int(sys.argv[5])
M = int(sys.argv[6])
target_dir = sys.argv[7]

assert sys.argv[2] in ["float", "double"]
assert sys.argv[3] in ["gpu", "cpu"]
assert model_name in ["numpyro", "vi", "max_lik", "stan"]

if use_double_precision:
    print("Using double precision")
    from jax.config import config

    config.update("jax_enable_x64", True)

import numpyro

if use_gpu:
    numpyro.set_platform("gpu")
else:
    numpyro.set_host_device_count(4)

import os
import time
import numpy as np
from ml_tools.patsy import create_formula
from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset_using_env_var,
    random_checklist_subset,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler

# from sdm_ml.checklist_level.ebird_joint_checklist_model_efficient import (
#     EBirdJointChecklistModel,
# )
from sdm_ml.checklist_level.ebird_joint_checklist_model import (
    EBirdJointChecklistModel as EBirdJointChecklistModelDesignMat,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model_numpyro import (
    EBirdJointChecklistModelNumpyro,
)
from sdm_ml.checklist_level.linear_checklist_model import LinearChecklistModel
from sdm_ml.checklist_level.ebird_joint_checklist_model_stan import (
    EBirdJointChecklistModelStan,
)


ebird_dataset = load_ebird_dataset_using_env_var()
train_set = ebird_dataset["train"]
bio_covs = [x for x in train_set.X_env.columns if "bio" in x]
train_covs = train_set.X_env

stan_model_path = (
    "/home/martin/projects/sdm_ml/sdm_ml/checklist_level/stan/checklist_model.stan"
)

# min_presences = 5
# n_species = 32
n_species = None

# Make a subset of the training checklists
np.random.seed(2)

if subset_size == -1:
    # Use full dataset
    subset_size = train_set.X_obs.shape[0]

subsetting_result = random_checklist_subset(
    train_set.X_obs.shape[0], train_set.env_cell_ids, subset_size
)

choice = subsetting_result["checklist_indices"]
species_counts = train_set.y_obs.iloc[choice].sum()
species_subset = species_counts[species_counts >= min_presences].index

if n_species is not None:
    species_subset = species_subset[
        np.random.choice(len(species_subset), size=n_species, replace=False)
    ]

test_set = ebird_dataset["test"]
test_covs = test_set.X_env
test_y = test_set.y_obs
test_cell_ids = test_set.env_cell_ids

# Check that all the required fields are present
for cur_field in ["time_of_day", "protocol_type"]:
    unique_test = test_set.X_obs[cur_field].unique()
    unique_train = train_set.X_obs.iloc[choice][cur_field].unique()
    print(cur_field, unique_test, unique_train)
    assert set(unique_test) == set(unique_train)

env_formula = create_formula(
    bio_covs,
    main_effects=True,
    quadratic_effects=True,
    interactions=False,
)

# Add on the other ones
to_add = [
    x
    for x in train_set.X_env.columns
    if "X" in x and x != "X" and x != "X12" or "dominant" in x
]
combined = "+".join(to_add)

train_covs = train_covs[bio_covs + to_add]
test_covs = test_covs[bio_covs + to_add]

# env_formula = env_formula + "+" + combined
env_formula = env_formula + "+" + "dominant_cover"

print(env_formula)

obs_formula = "protocol_type + daytimes_alt + log_duration_z + dominant_land_cover"

chain_method = "vectorized" if use_gpu else "parallel"
suffix = "_gpu" if use_gpu else "_cpu"
suffix = suffix + "_double" if use_double_precision else suffix + "_float"

models = {
    "numpyro"
    + suffix: EBirdJointChecklistModelNumpyro(
        env_formula,
        obs_formula,
        n_draws=1000,
        n_tune=1000,
        thinning=4,
        chain_method=chain_method,
    ),
    "vi": EBirdJointChecklistModelDesignMat(
        env_formula=env_formula, obs_formula=obs_formula, M=M
    ),
    "max_lik": LinearChecklistModel(env_formula, obs_formula),
    "stan": EBirdJointChecklistModelStan(stan_model_path, env_formula, obs_formula),
}

# Fit only model indicated
models = {
    x: y for x, y in models.items() if x == model_name or x == model_name + suffix
}

os.makedirs(target_dir, exist_ok=True)

for cur_model_name, model in models.items():

    X_env = train_covs.iloc[subsetting_result["env_cell_indices"]]
    X_checklist = train_set.X_obs.iloc[choice][
        [
            "protocol_type",
            "log_duration",
            "time_of_day",
            "dominant_land_cover",
            "daytimes_alt",
        ]
    ]
    y_checklist = train_set.y_obs[species_subset].iloc[choice]
    checklist_cell_ids = subsetting_result["checklist_cell_ids"]

    scaler = StandardScaler()
    to_scale = bio_covs
    not_scaled = to_add
    scaled_bio_covs = scaler.fit_transform(X_env[bio_covs])
    X_env_scaled = pd.concat(
        [
            pd.DataFrame(scaled_bio_covs, index=X_env.index, columns=bio_covs),
            X_env[not_scaled],
        ],
        axis=1,
    )

    # Standardise log duration
    log_duration_mean = X_checklist["log_duration"].mean()
    log_duration_std = X_checklist["log_duration"].std()

    X_checklist["log_duration_z"] = (
        X_checklist["log_duration"] - log_duration_mean
    ) / log_duration_std

    X_env.to_csv(os.path.join(target_dir, "X_env.csv"))
    X_checklist.to_csv(os.path.join(target_dir, "X_checklist.csv"))
    y_checklist.to_csv(os.path.join(target_dir, "y_checklist.csv"))
    np.savez(os.path.join("cell_ids"), checklist_cell_ids)

    assert not np.any(X_env.isnull().values)
    assert not np.any(X_checklist.isnull().values)
    assert not np.any(y_checklist.isnull().values)
    assert not np.any(np.isnan(checklist_cell_ids))

    start_time = time.time()
    model.fit(
        X_env=X_env_scaled,
        X_checklist=X_checklist,
        y_checklist=y_checklist,
        checklist_cell_ids=checklist_cell_ids,
    )
    runtime = time.time() - start_time

    # Evaluate on test set
    rel_y = test_y[species_subset]
    rel_covs = test_covs.loc[test_cell_ids]

    scaled_bio_covs = scaler.transform(rel_covs[bio_covs])
    rel_covs_scaled = pd.concat(
        [
            pd.DataFrame(scaled_bio_covs, index=rel_covs.index, columns=bio_covs),
            rel_covs[not_scaled],
        ],
        axis=1,
    )

    cur_target_dir = target_dir + cur_model_name + "/"
    model.save_model(cur_target_dir)
    print(runtime, file=open(cur_target_dir + "runtime.txt", "w"))

    pred_pres_prob = model.predict_marginal_probabilities_direct(rel_covs_scaled)
    pred_pres_prob.to_csv(os.path.join(cur_target_dir, "pres_preds.csv"))

    X_obs_test = test_set.X_obs
    X_obs_test["log_duration_z"] = (
        X_obs_test["log_duration"] - log_duration_mean
    ) / log_duration_std

    pred_obs_prob = model.predict_marginal_probabilities_obs(
        rel_covs_scaled, X_obs_test
    )
    pred_obs_prob.to_csv(os.path.join(cur_target_dir, "obs_preds.csv"))

    rel_y.to_csv(os.path.join(cur_target_dir, "y_t.csv"))

    species_counts.to_csv(os.path.join(cur_target_dir, "species_counts.csv"))
