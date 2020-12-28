import os
import numpy as np
from sdm_ml.checklist_level.checklist_dataset import load_ebird_dataset_using_env_var
from sdm_ml.checklist_level.ebird_joint_checklist_model_efficient import (
    EBirdJointChecklistModel,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model_stan import (
    EBirdJointChecklistModelStan,
)
from sdm_ml.checklist_level.linear_checklist_model import LinearChecklistModel
from ml_tools.patsy import create_formula
import time


ebird_dataset = load_ebird_dataset_using_env_var()
train_set = ebird_dataset["train"]
bio_covs = [x for x in train_set.X_env.columns if "bio" in x]
train_covs = train_set.X_env[bio_covs]

subset_size = int(1e3)
min_presences = 5
n_species = 8

# Make a subset of the training checklists
np.random.seed(2)
choice = np.random.choice(train_set.X_obs.shape[0], size=subset_size, replace=False)

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

env_formula = create_formula(
    bio_covs, main_effects=True, quadratic_effects=True, interactions=False,
)
obs_formula = "protocol_type + protocol_type:log_duration + time_of_day + log_duration"

models = {
    "stan_checklist_model": EBirdJointChecklistModelStan(
        "/home/martin/projects/sdm_ml/sdm_ml/checklist_level/stan/checklist_model.stan"
    ),
    "checklist_model_vi": EBirdJointChecklistModel(M=15, env_interactions=False),
    "linear_checklist_max_lik": LinearChecklistModel(env_formula, obs_formula),
}

target_dir = "./evaluations/stan_tests/"

for cur_model_name, model in models.items():

    # Subset properly
    # TODO: Make sure this subsetting routine is correct. Write a test.
    checklist_cell_ids = train_set.env_cell_ids[choice]
    cells_used = sorted(np.unique(checklist_cell_ids))
    train_cov_subset = train_covs.iloc[cells_used]
    new_cell_ids = np.arange(len(cells_used))

    mapping = {x: y for x, y in zip(cells_used, new_cell_ids)}
    cell_id_subset = np.array([mapping[x] for x in checklist_cell_ids])

    start_time = time.time()
    model.fit(
        X_env=train_cov_subset,
        X_checklist=train_set.X_obs.iloc[choice],
        y_checklist=train_set.y_obs[species_subset].iloc[choice],
        checklist_cell_ids=cell_id_subset,
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
