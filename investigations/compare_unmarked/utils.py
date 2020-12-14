from sdm_ml.checklist_level.checklist_dataset import (
    load_ebird_dataset_using_env_var,
    get_arrays_for_fitting,
)
import numpy as np
from os.path import join


def fetch_data(species_name="Cardinalis cardinalis"):

    ebird_dataset = load_ebird_dataset_using_env_var()

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
