from .checklist_model import ChecklistModel
import numpy as np
from typing import Callable
import pandas as pd


class TGBChecklistModel(ChecklistModel):
    def __init__(self, presence_absence_model):

        self.pa_model = presence_absence_model

    def fit(
        self,
        X_env_cell: np.ndarray,
        X_checklist: Callable[[str], pd.DataFrame],
        y_checklist: Callable[[str], np.ndarray],
        checklist_cell_ids: Callable[[str], np.ndarray],
        species_names: np.ndarray,
    ):

        y_full = list()
        cell_ids = None

        print("Preparing data...")
        for cur_species in species_names:

            cur_y = y_checklist(cur_species)
            cur_cell_ids = checklist_cell_ids(cur_species)
            counts_by_cell = np.bincount(cur_cell_ids, weights=cur_y)
            is_present = counts_by_cell > 0
            y_full.append(is_present)

            if cell_ids is None:
                cell_ids = cur_cell_ids
            else:
                assert np.all(cur_cell_ids == cell_ids)

        cell_ids = np.arange(0, np.max(cell_ids + 1))
        y_full = np.stack(y_full, axis=1)
        X_env = X_env_cell[cell_ids]
        print("Done.")

        print("Fitting PA Model...")
        # Fit the presence-absence model
        self.pa_model.fit(X_env, y_full)

    def predict_log_marginal_probabilities(self, X: np.ndarray) -> np.ndarray:

        return self.pa_model.predict_log_marginal_probabilities(X)

    def calculate_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        return self.pa_model.calculate_log_likelihood(X, y)

    def save_model(self, target_folder: str) -> None:

        self.pa_model.save_model(target_folder)
