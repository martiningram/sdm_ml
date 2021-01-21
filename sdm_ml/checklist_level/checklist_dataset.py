# This class is to handle datasets arising from checklists.
from typing import NamedTuple, Dict, Callable, List
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from ml_tools.paths import base_name_from_path
from ml_tools.modelling import remove_correlated_variables
from sklearn.preprocessing import LabelEncoder
import os


class ChecklistData(NamedTuple):

    # Environmental covariates
    X_env: pd.DataFrame

    # Checklist data
    X_obs: pd.DataFrame

    # Presence/absence at checklist level
    y_obs: pd.DataFrame

    # Environment cells to match the entries in X_obs
    env_cell_ids: np.ndarray


def load_ebird_dataset_using_env_var(
    train_folds=[1, 2, 3],
    test_folds=[4],
    drop_nan_cells=True,
):

    ebird_dataset = load_ebird_dataset(
        join(os.environ["EBIRD_DATA_PATH"], "checklists_with_folds.csv"),
        join(os.environ["EBIRD_DATA_PATH"], "raster_cell_covs.csv"),
        join(os.environ["EBIRD_DATA_PATH"], "all_pa.csv"),
    )

    return ebird_dataset


def load_ebird_dataset(
    ebird_checklist_file,
    covariates_file,
    species_pa_file,
    train_folds=[1, 2, 3],
    test_folds=[4],
):

    sampling_data = pd.read_csv(ebird_checklist_file, index_col=0)
    species_pa_df = pd.read_csv(species_pa_file, index_col=0)
    covariates = pd.read_csv(covariates_file, index_col=0)

    # Drop NaN cells and their checklists:
    nan_cells = covariates["cell"][covariates.isnull().any(axis=1)]
    covariates = covariates.dropna()
    sampling_data = sampling_data[~sampling_data["cell_id"].isin(nan_cells)].set_index(
        "checklist_id"
    )

    # Encode the environment cells
    encoder = LabelEncoder()
    encoder.fit(covariates["cell"])

    covariates["cell"] = encoder.transform(covariates["cell"])
    covariates = covariates.sort_values("cell").set_index("cell")
    sampling_data["cell_id"] = encoder.transform(sampling_data["cell_id"])
    species_pa_df = species_pa_df.loc[sampling_data.index]

    sampling_data = add_derived_covariates(sampling_data)

    def split_data(sampling_data, pa_df, folds):

        cur_relevant = sampling_data["fold_id"].isin(folds)
        return sampling_data[cur_relevant], pa_df[cur_relevant]

    train_checklists, train_y = split_data(sampling_data, species_pa_df, train_folds)
    test_checklists, test_y = split_data(sampling_data, species_pa_df, test_folds)

    # Ditch overly-correlated covariates
    all_cov_names = [x for x in covariates.columns if "bio" in x]
    other_names = [x for x in covariates.columns if x not in all_cov_names]
    to_keep = list(remove_correlated_variables(covariates[all_cov_names])) + other_names
    covariates = covariates[to_keep]

    train_data = ChecklistData(
        X_env=covariates,
        X_obs=train_checklists,
        y_obs=train_y,
        env_cell_ids=train_checklists["cell_id"].values,
    )

    test_data = ChecklistData(
        X_env=covariates,
        X_obs=test_checklists,
        y_obs=test_y,
        env_cell_ids=test_checklists["cell_id"].values,
    )

    return {"train": train_data, "test": test_data}


def random_cell_subset(
    n_cells, numeric_checklist_cell_ids, n_cells_to_pick=1000, seed=2
):

    np.random.seed(seed)

    # We assume here that there are cells numbered 1, ... n_cells.
    # First, pick the cells we are keeping:
    # These will be the indices for the array.
    picked = sorted(np.random.choice(n_cells, size=n_cells_to_pick, replace=False))

    # Create the mask to subset the checklist data:
    checklist_mask = np.isin(numeric_checklist_cell_ids, picked)

    # Now we also need new checklist ids
    old_corresponding_cells = numeric_checklist_cell_ids[checklist_mask]
    old_to_new_encoder = LabelEncoder()
    old_to_new_encoder.fit(picked)
    new_corresponding_cells = old_to_new_encoder.transform(old_corresponding_cells)

    return picked, checklist_mask, new_corresponding_cells


def random_checklist_subset(
    n_checklists, checklist_cell_ids, n_checklists_to_pick, seed=2
):

    np.random.seed(seed)

    picked_checklists = np.random.choice(
        n_checklists, size=n_checklists_to_pick, replace=False
    )

    picked_ids = checklist_cell_ids[picked_checklists]

    cells_used = sorted(np.unique(picked_ids))

    # Map from old to new
    mapping = {x: i for i, x in enumerate(cells_used)}
    new_ids = np.array([mapping[x] for x in picked_ids])

    return {
        "env_cell_indices": cells_used,
        "checklist_cell_ids": new_ids,
        "checklist_indices": picked_checklists,
    }


def add_derived_covariates(ebird_obs_df):

    obs_covs = ebird_obs_df

    obs_covs["log_duration"] = np.log(obs_covs["duration_minutes"])

    # TODO: This logic should perhaps be moved somewhere more central?
    hour_of_day = (
        obs_covs["time_observations_started"].str.split(":").str.get(0).astype(int)
    )

    time_of_day = np.select(
        [
            (hour_of_day >= 5) & (hour_of_day < 12),
            (hour_of_day > 12) & (hour_of_day < 21),
        ],
        ["morning", "afternoon/evening"],
        default="night",
    )

    obs_covs["time_of_day"] = time_of_day
    fine_str = (
        ((hour_of_day // 3) * 3).astype(str)
        + "-"
        + ((hour_of_day // 3) * 3 + 3).astype(str)
    )
    obs_covs["time_of_day_fine"] = fine_str

    obs_covs["protocol_type"] = obs_covs["protocol_type"].str.replace(
        "Traveling - Property Specific", "Traveling"
    )

    return obs_covs


if __name__ == "__main__":

    ebird_dataset = load_ebird_dataset_using_env_var()

    import ipdb

    ipdb.set_trace()
