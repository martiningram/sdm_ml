import numpy as np
from .checklist_dataset import get_arrays_for_fitting


def get_ebird_test_data(ebird_dataset, species_names):
    # TODO: Maybe move to the other checklist stuff

    # TODO: This could be improved
    arrays = get_arrays_for_fitting(ebird_dataset, species_names[0])
    test_obs_covs = ebird_dataset.test_checklist_data.observer_covariates
    covs = ebird_dataset.cell_data.cell_covariates[arrays["env_covariates"].columns]
    covs = covs.set_index(ebird_dataset.cell_data.cell_id)
    test_cell_ids = ebird_dataset.test_checklist_data.cell_id
    test_covs = covs.loc[test_cell_ids]

    for cur_species in species_names:

        species_data = ebird_dataset.get_species_data(cur_species)

        presence_lookup = {
            x: y for x, y in zip(species_data.checklist_id, species_data.was_observed)
        }

        test_y = np.array(
            [presence_lookup[x] for x in ebird_dataset.test_checklist_data.checklist_id]
        )

    return test_covs, test_obs_covs, test_y
