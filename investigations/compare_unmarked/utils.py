from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset,
    get_arrays_for_fitting,
)
import numpy as np


def fetch_data(species_name="Cardinalis cardinalis"):

    # TODO: Use env variable
    ebird_dataset = load_ebird_dataset(
        "/home/martin/data/ebird-basic-dataset/scripts/checklists_with_folds.csv",
        "/home/martin/data/ebird-basic-dataset/scripts/raster_cell_covs.csv",
        "/home/martin/data/ebird-basic-dataset/june_2019/species_separated/",
    )

    arrays = get_arrays_for_fitting(ebird_dataset, species_name)

    obs_covs = arrays["checklist_data"].observer_covariates
    obs_covs["cell_id"] = arrays["numeric_checklist_cell_ids"]
    obs_covs["log_duration"] = np.log(obs_covs["duration_minutes"])

    # TODO: This logic should perhaps be moved somewhere more central?
    hour_of_day = (
        arrays["checklist_data"]
        .observer_covariates["time_observations_started"]
        .str.split(":")
        .str.get(0)
        .astype(int)
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
    obs_covs["protocol_type"] = obs_covs["protocol_type"].str.replace(
        "Traveling - Property Specific", "Traveling"
    )

    return obs_covs, arrays
