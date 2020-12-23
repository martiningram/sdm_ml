import os
import numpy as np
from sdm_ml.checklist_level.checklist_dataset import load_ebird_dataset_using_env_var
from sdm_ml.checklist_level.ebird_joint_checklist_model_efficient import (
    EBirdJointChecklistModel,
)
from sdm_ml.checklist_level.linear_checklist_model import LinearChecklistModel
from ml_tools.patsy import create_formula
import time


ebird_dataset = load_ebird_dataset_using_env_var()
train_set = ebird_dataset["train"]
bio_covs = [x for x in train_set.X_env.columns if "bio" in x]
train_covs = train_set.X_env[bio_covs]

subset_size = int(1e5)

# Make a subset of the training checklists
np.random.seed(2)
choice = np.random.choice(train_set.X_obs.shape[0], size=int(1e5), replace=False)

test_set = ebird_dataset["test"]
test_covs = test_set.X_env[bio_covs]
test_y = test_set.y_obs
test_cell_ids = test_set.env_cell_ids

env_formula = create_formula(
    bio_covs, main_effects=True, quadratic_effects=True, interactions=False,
)
obs_formula = "protocol_type + protocol_type:log_duration + time_of_day + log_duration"

models = {
    "checklist_model_vi": EBirdJointChecklistModel(M=15, env_interactions=False),
    "linear_checklist_max_lik": LinearChecklistModel(env_formula, obs_formula),
}

target_dir = "./evaluations/revamp/"

for cur_model_name, model in models.items():

    start_time = time.time()
    model.fit(
        X_env=train_covs,
        X_checklist=train_set.X_obs.iloc[choice],
        y_checklist=train_set.y_obs.iloc[choice].iloc[:, 4:8],
        checklist_cell_ids=train_set.env_cell_ids[choice],
    )
    runtime = time.time() - start_time

    # Evaluate on test set
    rel_y = test_y.iloc[:, 4:8]
    rel_covs = test_covs.loc[test_cell_ids]

    cur_target_dir = target_dir + cur_model_name + "/"
    model.save_model(cur_target_dir)

    pred_obs_prob = model.predict_marginal_probabilities_obs(rel_covs, test_set.X_obs)
    pred_obs_prob.to_csv(os.path.join(cur_target_dir, "obs_preds.csv"))

    pred_pres_prob = model.predict_marginal_probabilities_direct(train_covs)
    pred_pres_prob.to_csv(os.path.join(cur_target_dir, "pres_preds.csv"))

    rel_y.to_csv(os.path.join(cur_target_dir, "y_t.csv"))

    print(runtime, file=open(cur_target_dir + "runtime.txt", "w"))
