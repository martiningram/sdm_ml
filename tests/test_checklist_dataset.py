from sdm_ml.checklist_level.checklist_dataset import (
    random_cell_subset,
    random_checklist_subset,
)
import numpy as np


def generate_random_data(n_cells, n_checklists):

    # Make up some data
    X_env = np.random.randn(n_cells, 3)
    X_checklist = np.random.randn(n_checklists, 2)
    cell_ids = np.random.choice(n_cells, size=n_checklists)

    return X_env, X_checklist, cell_ids


def test_random_cell_subset():

    n_cells = 20
    n_checklists = 100

    X_env, X_checklist, cell_ids = generate_random_data(n_cells, n_checklists)

    subset_size = 5

    picked, checklist_mask, new_corresponding_cells = random_cell_subset(
        n_cells, cell_ids, n_cells_to_pick=subset_size
    )

    X_env_subset = X_env[picked]
    X_checklist_subset = X_checklist[checklist_mask]

    for i, cur_picked in enumerate(picked):

        # Make sure environment still matches.
        old_X_env = X_env[cur_picked]
        new_X_env = X_env_subset[i]
        assert np.allclose(old_X_env, new_X_env)

        # Make sure checklist data matches
        old_X_checklist = X_checklist[cell_ids == cur_picked]
        new_X_checklist = X_checklist_subset[new_corresponding_cells == i]
        assert np.allclose(old_X_checklist, new_X_checklist)


def test_random_checklist_subset():

    n_cells = 20
    n_checklists = 100

    X_env, X_checklist, cell_ids = generate_random_data(n_cells, n_checklists)

    subset_size = 20

    subset_result = random_checklist_subset(n_checklists, cell_ids, subset_size)

    X_env_subset = X_env[subset_result["env_cell_indices"]]
    X_checklist_subset = X_checklist[subset_result["checklist_indices"]]
    checklist_indices_subset = subset_result["checklist_cell_ids"]

    for i, cur_picked in enumerate(subset_result["checklist_indices"]):

        old_X_checklist = X_checklist[cur_picked]
        old_cell_id = cell_ids[cur_picked]
        old_X_env = X_env[old_cell_id]

        new_X_checklist = X_checklist_subset[i]
        new_cell_id = checklist_indices_subset[i]
        new_X_env = X_env_subset[new_cell_id]

        # Make sure these match
        assert np.all(old_X_env == new_X_env)
        assert np.all(old_X_checklist == new_X_checklist)
