# This class is to handle datasets arising from checklists.
from typing import NamedTuple, Dict, Callable, List
import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from ml_tools.paths import base_name_from_path
from ml_tools.modelling import remove_correlated_variables
from sklearn.preprocessing import LabelEncoder


class ChecklistLevelData(NamedTuple):

    # This links each checklist to a cell
    cell_id: np.ndarray

    # The checklist id
    checklist_id: np.ndarray

    # These are covariates recorded per checklist such as duration
    observer_covariates: pd.DataFrame

    # The location
    coords: pd.DataFrame

    # Whatever else is recorded
    other_info: pd.DataFrame
    coord_code: int = 4326


class SpeciesSightingData(NamedTuple):

    species_name: str
    was_observed: np.ndarray
    checklist_id: np.ndarray
    observation_count: np.ndarray = None


class CellLevelData(NamedTuple):

    cell_id: np.ndarray
    cell_covariates: pd.DataFrame
    cell_lat_lon: pd.DataFrame = None


class ChecklistDataset(NamedTuple):

    train_checklist_data: ChecklistLevelData
    test_checklist_data: ChecklistLevelData

    cell_data: CellLevelData
    species_names: np.ndarray

    # This is a callable because, for some datasets, it might be expensive to
    # load all the (zero-filled) sightings at once.
    get_species_data: Callable[[str], SpeciesSightingData]


def get_arrays_for_fitting(
    checklist_data: ChecklistDataset,
    species_name: str,
    drop_correlated_covariates=True,
):

    # Get sightings
    species_data = checklist_data.get_species_data(species_name)

    sightings_lookup = {
        checklist_id: is_present
        for checklist_id, is_present in zip(
            species_data.checklist_id, species_data.was_observed
        )
    }

    checklist_sightings = [
        sightings_lookup.get(cur_cell_id, None)
        for cur_cell_id in checklist_data.train_checklist_data.checklist_id
    ]

    checklist_cell_ids = checklist_data.train_checklist_data.cell_id

    all_cell_ids = checklist_data.cell_data.cell_id
    rel_cell_ids = set(all_cell_ids) & set(checklist_cell_ids)

    assert len(set(checklist_cell_ids) - set(all_cell_ids)) == 0

    is_relevant = [x in rel_cell_ids for x in all_cell_ids]
    rel_covariates = checklist_data.cell_data.cell_covariates[is_relevant]
    rel_cell_ids = checklist_data.cell_data.cell_id[is_relevant]

    encoder = LabelEncoder()
    numeric_cell_ids = encoder.fit_transform(rel_cell_ids)

    # Ensure things are sorted
    assert all(
        numeric_cell_ids[i] <= numeric_cell_ids[i + 1]
        for i in range(len(numeric_cell_ids) - 1)
    ), "Numeric cell ids are not sorted."

    if drop_correlated_covariates:
        to_keep = remove_correlated_variables(rel_covariates)
        rel_covariates = rel_covariates[to_keep]

    return {
        "checklist_data": checklist_data.train_checklist_data,
        "env_covariates": rel_covariates,
        "is_present": np.array(checklist_sightings),
        "numeric_env_cell_ids": numeric_cell_ids,
        "numeric_checklist_cell_ids": encoder.transform(
            checklist_data.train_checklist_data.cell_id
        ),
    }


def load_ebird_dataset(
    ebird_checklist_file,
    covariates_file,
    ebird_species_dir,
    train_folds=[1, 2, 3],
    test_folds=[4],
    drop_nan_cells=True,
):
    # The checklist file contains the checklist-level information; the species
    # directory contains csvs, one for each bird.
    sampling_data = pd.read_csv(ebird_checklist_file, index_col=0)
    species_files = glob(join(ebird_species_dir, "*.csv"))
    species_names = np.array([base_name_from_path(x) for x in species_files])
    covariates = pd.read_csv(covariates_file, index_col=0)

    # NOTE: This is pretty crucial to ensure that the numerical labels later
    # match up.
    covariates = covariates.sort_values("cell").reset_index(drop=True)

    assert drop_nan_cells, "Currently expects NaN cells to be dropped"

    nan_cells = covariates["cell"][covariates.isnull().any(axis=1)]
    covariates = covariates.dropna()

    species_file_lookup = {x: y for x, y in zip(species_names, species_files)}

    def get_species_data(species_name):

        cur_file = species_file_lookup[species_name]
        loaded = pd.read_csv(cur_file, index_col=0)

        checklist_id = loaded["checklist_id"].values
        was_observed = loaded["species_observed"].values
        observation_count = loaded["observation_count"].values

        return SpeciesSightingData(
            species_name, was_observed, checklist_id, observation_count
        )

    def get_checklist_data(sampling_data, drop_bbs=True):

        if drop_nan_cells:
            sampling_data = sampling_data[~sampling_data["cell_id"].isin(nan_cells)]

        if drop_bbs:
            sampling_data = sampling_data[
                ~sampling_data["locality"].str.contains("BBS")
            ]

        # Put together the checklist-level data
        checklist_id = sampling_data["checklist_id"].values

        observer_covariates = sampling_data[
            [
                "protocol_type",
                "protocol_code",
                "project_code",
                "duration_minutes",
                "effort_distance_km",
                "effort_area_ha",
                "number_observers",
                "all_species_reported",
                "observation_date",
                "time_observations_started",
                "observer_id",
                "sampling_event_identifier",
            ]
        ]

        other_info = sampling_data[
            [x for x in sampling_data.columns if x not in observer_covariates.columns]
        ]

        lat_lon = sampling_data[["X", "Y"]]
        coord_code = 3968

        cell_id = sampling_data["cell_id"]

        return ChecklistLevelData(
            cell_id.values,
            checklist_id,
            observer_covariates,
            lat_lon,
            other_info,
            coord_code,
        )

    train_sampling_data = sampling_data[sampling_data["fold_id"].isin(train_folds)]
    test_sampling_data = sampling_data[sampling_data["fold_id"].isin(test_folds)]

    train_checklist_data = get_checklist_data(train_sampling_data, drop_bbs=True)
    test_checklist_data = get_checklist_data(test_sampling_data, drop_bbs=False)

    cell_data = CellLevelData(
        cell_id=covariates["cell"].values,
        cell_covariates=covariates[
            [x for x in covariates.columns if x not in ["cell", "x", "y"]]
        ],
        cell_lat_lon=covariates[["x", "y"]],
    )

    return ChecklistDataset(
        train_checklist_data=train_checklist_data,
        test_checklist_data=test_checklist_data,
        cell_data=cell_data,
        species_names=species_names,
        get_species_data=get_species_data,
    )


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
