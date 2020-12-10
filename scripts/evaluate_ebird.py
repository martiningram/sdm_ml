import os
from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset,
    get_arrays_for_fitting,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model_efficient import EBirdJointChecklistModel
from sdm_ml.dataset import load_bbs_dataset_2019
from sdm_ml.evaluation import compute_and_save_results_for_evaluation
from sdm_ml.dataset import SpeciesData
import pandas as pd
import time


ebird_dataset = load_ebird_dataset(
    "/home/martin/data/scripts/checklists_with_folds.csv",
    "/home/martin/data/scripts/raster_cell_covs.csv",
    "/home/martin/data/species_separated"
)

model = EBirdJointChecklistModel(env_interactions=False)

bbs_2019 = load_bbs_dataset_2019(
    "/home/martin/data/scripts/bbs_with_folds.csv",
    "/home/martin/data/scripts/raster_cell_covs.csv"
)

# Fetch arrays for one species
cur_species = "Larus argentatus"
arrays = get_arrays_for_fitting(ebird_dataset, cur_species)
X_env_cell = arrays["env_covariates"]


# Define functions for fitting
def X_checklist(species_name):

    arrays = get_arrays_for_fitting(ebird_dataset, species_name)
    X_obs = arrays["checklist_data"].observer_covariates
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

start_time = time.time()
model.fit(
    X_env_cell=X_env_cell,
    X_checklist=X_checklist,
    y_checklist=y_checklist,
    checklist_cell_ids=checklist_cell_ids,
    species_names=species_names,
)
runtime = time.time() - start_time

test_set = bbs_2019["test"]

test_set_dict = test_set._asdict()

test_set_dict["outcomes"] = test_set_dict["outcomes"][species_names]
test_set_dict["covariates"] = pd.DataFrame(
    test_set_dict["covariates"][arrays["env_covariates"].columns]
)

test_set_amended = SpeciesData(**test_set_dict)

target_dir = "./evaluations/hierarchical_checklist_model_advi_rerun_no_inter_no_env_hierarchy/"

compute_and_save_results_for_evaluation(
    test_set_amended, model, target_dir)

print(runtime, file=open(target_dir + 'runtime.txt', "w"))
