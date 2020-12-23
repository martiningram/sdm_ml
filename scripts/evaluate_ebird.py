import os
import numpy as np
from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset,
    get_arrays_for_fitting,
    add_derived_covariates,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model import EBirdJointChecklistModel
from sdm_ml.checklist_level.linear_checklist_model import LinearChecklistModel
from sdm_ml.dataset import load_bbs_dataset_2019
from sdm_ml.evaluation import compute_and_save_results_for_evaluation
from sdm_ml.dataset import SpeciesData
import pandas as pd
import time
from os.path import join
from sdm_ml.checklist_level.tgb_checklist_model import TGBChecklistModel
from sdm_ml.linear_model_advi import LinearModelADVI
from sdm_ml.scikit_model import ScikitModel
from ml_tools.patsy import create_formula


ebird_dataset = load_ebird_dataset(
    join(os.environ["EBIRD_DATA_PATH"], "checklists_with_folds.csv"),
    join(os.environ["EBIRD_DATA_PATH"], "raster_cell_covs.csv"),
    os.environ["EBIRD_SPECIES_DATA_PATH"],
)


# Fetch arrays for one species
cur_species = "Larus argentatus"
arrays = get_arrays_for_fitting(ebird_dataset, cur_species)
X_env_cell = arrays["env_covariates"]

n_checklists = arrays["checklist_data"].observer_covariates.shape[0]

# checklist_subset = np.random.choice(
#     n_checklists,
#     size=50000, replace=False
# )

# No subset
checklist_subset = np.arange(n_checklists)


# Define functions for fitting
def X_checklist(species_name):

    arrays = get_arrays_for_fitting(ebird_dataset, species_name)
    X_obs = arrays["checklist_data"].observer_covariates
    X_obs = add_derived_covariates(X_obs)
    return X_obs.iloc[checklist_subset]


y_checklist = lambda species_name: get_arrays_for_fitting(ebird_dataset, species_name)[
    "is_present"
][checklist_subset]

checklist_cell_ids = lambda species_name: get_arrays_for_fitting(
    ebird_dataset, species_name
)["numeric_checklist_cell_ids"][checklist_subset]

species_names = sorted(
    list(set(ebird_dataset.species_names) & (set(bbs_2019["train"].outcomes.columns)))
)

# species_names = species_names[:4]
env_formula = create_formula(
    arrays["env_covariates"].columns,
    main_effects=True,
    quadratic_effects=True,
    interactions=False,
)
obs_formula = "protocol_type + protocol_type:log_duration + time_of_day + log_duration"

models = {
    "checklist_model_vi_design_mat": EBirdJointChecklistModel(env_interactions=False),
    # "linear_checklist_max_lik": LinearChecklistModel(env_formula, obs_formula),
}

for cur_model_name, model in models.items():

    start_time = time.time()
    model.fit(
        X_env_cell=X_env_cell,
        X_checklist=X_checklist,
        y_checklist=y_checklist,
        checklist_cell_ids=checklist_cell_ids,
        species_names=species_names,
    )
    runtime = time.time() - start_time

    print(runtime, file=open(cur_target_dir + "runtime.txt", "w"))
