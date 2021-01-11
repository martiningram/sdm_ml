from jax.config import config

config.update("jax_enable_x64", True)
import numpyro

numpyro.set_platform("gpu")
import os
import time
import numpy as np
from ml_tools.patsy import create_formula
from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset_using_env_var,
    random_checklist_subset,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model_efficient import (
    EBirdJointChecklistModel,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model_numpyro import (
    EBirdJointChecklistModelNumpyro,
)
from sdm_ml.checklist_level.linear_checklist_model import LinearChecklistModel


ebird_dataset = load_ebird_dataset_using_env_var()
train_set = ebird_dataset["train"]
bio_covs = [x for x in train_set.X_env.columns if "bio" in x]
train_covs = train_set.X_env[bio_covs]

subset_size = int(50000)
# subset_size = int(1000)
# subset_size = train_set.X_obs.shape[0]
min_presences = 5
# n_species = 32
n_species = None

# Make a subset of the training checklists
np.random.seed(2)
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
test_covs = test_set.X_env[bio_covs]
test_y = test_set.y_obs
test_cell_ids = test_set.env_cell_ids

# Check that all the required fields are present
for cur_field in ["time_of_day", "protocol_type"]:
    unique_test = test_set.X_obs[cur_field].unique()
    unique_train = train_set.X_obs.iloc[choice][cur_field].unique()
    print(cur_field, unique_test, unique_train)
    assert set(unique_test) == set(unique_train)

env_formula = create_formula(
    bio_covs, main_effects=True, quadratic_effects=True, interactions=False,
)
obs_formula = "protocol_type + protocol_type:log_duration + time_of_day + log_duration"

models = {
    "checklist_model_numpyro": EBirdJointChecklistModelNumpyro(
        env_formula, obs_formula, n_draws=1000, n_tune=1000, thinning=4
    ),
    "checklist_model_vi": EBirdJointChecklistModel(M=25, env_interactions=False),
    "linear_checklist_max_lik": LinearChecklistModel(env_formula, obs_formula),
}

target_dir = "./evaluations/numpyro_comparison_rerun_1k_double/"

for cur_model_name, model in models.items():

    X_env = train_covs.iloc[subsetting_result["env_cell_indices"]]
    X_checklist = train_set.X_obs.iloc[choice][
        [
            "protocol_type",
            "log_duration",
            "time_of_day",
            "time_observations_started",
            "duration_minutes",
        ]
    ]
    y_checklist = train_set.y_obs[species_subset].iloc[choice]
    checklist_cell_ids = subsetting_result["checklist_cell_ids"]

    assert not np.any(X_env.isnull().values)
    assert not np.any(X_checklist.isnull().values)
    assert not np.any(y_checklist.isnull().values)
    assert not np.any(np.isnan(checklist_cell_ids))

    start_time = time.time()
    model.fit(
        X_env=X_env,
        X_checklist=X_checklist,
        y_checklist=y_checklist,
        checklist_cell_ids=checklist_cell_ids,
    )
    runtime = time.time() - start_time

    # Evaluate on test set
    rel_y = test_y[species_subset]
    rel_covs = test_covs.loc[test_cell_ids]

    cur_target_dir = target_dir + cur_model_name + "/"
    model.save_model(cur_target_dir)

    pred_pres_prob = model.predict_marginal_probabilities_direct(train_covs)
    pred_pres_prob.to_csv(os.path.join(cur_target_dir, "pres_preds.csv"))

    pred_obs_prob = model.predict_marginal_probabilities_obs(rel_covs, test_set.X_obs)
    pred_obs_prob.to_csv(os.path.join(cur_target_dir, "obs_preds.csv"))

    rel_y.to_csv(os.path.join(cur_target_dir, "y_t.csv"))

    species_counts.to_csv(os.path.join(cur_target_dir, "species_counts.csv"))

    print(runtime, file=open(cur_target_dir + "runtime.txt", "w"))
