from sdm_ml.checklist_level.checklist_dataset import random_cell_subset
import numpy as np


def test_random_cell_subset():

    n_cells = 20
    n_checklists = 50

    # Make up some data
    X_env = np.random.randn(n_cells, 3)
    X_checklist = np.random.randn(n_checklists, 2)
    cell_ids = np.random.choice(n_cells, size=n_checklists)

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
