import os
from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset,
    get_arrays_for_fitting,
    add_derived_covariates,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model_efficient import (
    EBirdJointChecklistModel,
)
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


bbs_2019 = load_bbs_dataset_2019(
    join(os.environ["EBIRD_DATA_PATH"], "bbs_with_folds.csv"),
    join(os.environ["EBIRD_DATA_PATH"], "raster_cell_covs.csv"),
)

# Fetch arrays for one species
cur_species = "Larus argentatus"
arrays = get_arrays_for_fitting(ebird_dataset, cur_species)
X_env_cell = arrays["env_covariates"]


# Define functions for fitting
def X_checklist(species_name):

    arrays = get_arrays_for_fitting(ebird_dataset, species_name)
    X_obs = arrays["checklist_data"].observer_covariates
    X_obs = add_derived_covariates(X_obs)
    return X_obs


y_checklist = lambda species_name: get_arrays_for_fitting(ebird_dataset, species_name)[
    "is_present"
]

checklist_cell_ids = lambda species_name: get_arrays_for_fitting(
    ebird_dataset, species_name
)["numeric_checklist_cell_ids"]

species_names = sorted(
    list(set(ebird_dataset.species_names) & (set(bbs_2019["train"].outcomes.columns)))
)

# species_names = species_names[:4]

linear_advi_model = LinearModelADVI()
rf_model = ScikitModel(
    model_fun=lambda: ScikitModel.create_cross_validated_forest(
        len(arrays["env_covariates"].columns)
    )
)

env_formula = create_formula(
    arrays["env_covariates"].columns,
    main_effects=True,
    quadratic_effects=True,
    interactions=False,
)
obs_formula = "protocol_type + protocol_type:log_duration + time_of_day + log_duration"

models = {
    # "checklist_model_vi": EBirdJointChecklistModel(env_interactions=False),
    # "linear_model_vi_tgb_quadratic": TGBChecklistModel(linear_advi_model),
    # "rf_cv": TGBChecklistModel(rf_model),
    "linear_checklist_max_lik": LinearChecklistModel(env_formula, obs_formula),
}

test_set = bbs_2019["test"]

test_set_dict = test_set._asdict()

test_set_dict["outcomes"] = test_set_dict["outcomes"][species_names]
test_set_dict["covariates"] = pd.DataFrame(
    test_set_dict["covariates"][arrays["env_covariates"].columns]
)

test_set_amended = SpeciesData(**test_set_dict)

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

    cur_target_dir = f"./evaluations/comparison/{cur_model_name}/"

    compute_and_save_results_for_evaluation(test_set_amended, model, cur_target_dir)

    print(runtime, file=open(cur_target_dir + "runtime.txt", "w"))
