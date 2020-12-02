from sdm_ml.checklist_level.ebird_joint_checklist_model_efficient import (
    EBirdJointChecklistModel,
)
from sdm_ml.checklist_level.stan.ebird_joint_checklist_model_stan import (
    EBirdJointChecklistModelStan,
)
from sdm_ml.checklist_level.ebird_joint_checklist_model import (
    EBirdJointChecklistModel as EBirdJointChecklistModelSlower,
)
from sdm_ml.dataset import load_bbs_dataset_2019
import numpy as np
from sdm_ml.checklist_level.checklist_dataset import random_cell_subset
from ml_tools.utils import save_pickle_safely
import os
from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset,
    get_arrays_for_fitting,
)
import sys
from os import makedirs
from os.path import join
import pandas as pd
import time

subset_size = int(sys.argv[1])
n_species = int(sys.argv[2])
min_presence_count = int(sys.argv[3])
target_dir = sys.argv[4]

ebird_dataset = load_ebird_dataset(
    "/home/martin/data/ebird-basic-dataset/scripts/checklists_with_folds.csv",
    "/home/martin/data/ebird-basic-dataset/scripts/raster_cell_covs.csv",
    "/home/martin/data/ebird-basic-dataset/june_2019/species_separated/",
)

cur_species = "Anas crecca"

arrays = get_arrays_for_fitting(ebird_dataset, cur_species)

model = EBirdJointChecklistModelStan(
    "/home/martin/projects/sdm_ml/sdm_ml/checklist_level/stan/checklist_model.stan",
    env_interactions=False,
)

var_model_fast = EBirdJointChecklistModel(M=100, env_interactions=False)
var_model_slower = EBirdJointChecklistModelSlower(M=100, env_interactions=False)

bbs_2019 = load_bbs_dataset_2019(
    "/home/martin/data/ebird-basic-dataset/scripts/bbs_with_folds.csv",
    "/home/martin/data/ebird-basic-dataset/scripts/raster_cell_covs.csv",
)

X_env_cell = arrays["env_covariates"]

np.random.seed(3)

picks, checklist_mask, cell_ids = random_cell_subset(
    X_env_cell.shape[0],
    arrays["numeric_checklist_cell_ids"],
    n_cells_to_pick=subset_size,
)

rel_X_env = X_env_cell.iloc[picks]


def X_checklist(species_name):

    arrays = get_arrays_for_fitting(ebird_dataset, species_name)

    X_obs = arrays["checklist_data"].observer_covariates[checklist_mask]

    return X_obs


y_checklist = lambda species_name: get_arrays_for_fitting(ebird_dataset, species_name)[
    "is_present"
][checklist_mask].astype(int)
checklist_cell_ids = lambda species_name: cell_ids

all_species_names = sorted(
    list(set(ebird_dataset.species_names) & (set(bbs_2019["train"].outcomes.columns)))
)

assert n_species == 8

species_names = [
    "Pandion haliaetus",
    "Spizella passerina",
    "Mimus polyglottos",
    "Empidonax difficilis",
    "Ardea alba",
    "Gallinula galeata",
    "Phalacrocorax auritus",
    "Coccothraustes vespertinus",
]

# counts = {x: y_checklist(x).sum() for x in all_species_names}
# counts = pd.Series(counts)
# to_pick_from = counts[counts > min_presence_count].index
# species_names = np.random.choice(to_pick_from, size=n_species, replace=False)

fit_args = dict(
    X_env_cell=rel_X_env,
    X_checklist=X_checklist,
    y_checklist=y_checklist,
    checklist_cell_ids=checklist_cell_ids,
    species_names=species_names,
)

# Fit Stan model
start_time = time.time()
model.fit(**fit_args)
stan_time = time.time() - start_time

subdir_name = f"model_fit_n_cells_{subset_size}_n_species_{n_species}"

target_dir = join(target_dir, subdir_name)

makedirs(target_dir, exist_ok=True)

save_pickle_safely(model.fit_results.extract(), join(target_dir, "stan_samples.pkl"))
print(
    model.fit_results.stansummary(),
    file=open(join(target_dir, "stan_summary.txt"), "w"),
)
print(stan_time, file=open(join(target_dir, "stan_fit_time.txt"), "w"))

start_time = time.time()
var_model_fast.fit(**fit_args)
vi_fast_time = time.time() - start_time
print(vi_fast_time, file=open(join(target_dir, "vi_fast_fit_time.txt"), "w"))

var_model_fast.save_model(join(target_dir, "fast_variational"))

start_time = time.time()
var_model_slower.fit(**fit_args)
vi_slow_time = time.time() - start_time
print(vi_slow_time, file=open(join(target_dir, "vi_slow_fit_time.txt"), "w"))

var_model_slower.save_model(join(target_dir, "slower_variational"))
