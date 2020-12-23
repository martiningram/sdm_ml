import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
import pandas as pd


class ChecklistModel(ABC):
    @abstractmethod
    def fit(
        self,
        X_env_cell: pd.DataFrame,
        X_checklist: Callable[[str], pd.DataFrame],
        y_checklist: Callable[[str], np.ndarray],
        checklist_cell_ids: Callable[[str], np.ndarray],
        species_names: np.ndarray,
    ) -> None:
        """Fits a ChecklistModel.

        Args:
            X_env_cell: An NxM matrix, with N the number of cells and M the
                number of cell covariates. These are covariates assumed to drive
                the presence or absence of the species.
            X_checklist: An LxK matrix, with L the number of checklists and K
                the number of checklist covariates. These are covariates
                associated with each checklist such as duration.
            y_checklist: A function which, given a species name, returns the
                [zero-filled] presence or absence of that species for each of
                the L checklists.
            checklist_cell_ids: The cell ids for each of the checklists, so that
                entry i in this array gives the location of the corresponding
                covariates in X_env_cell.
            species_names: The names of all species to fit.
        """
        pass

    @abstractmethod
    def predict_marginal_probabilities_direct(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_marginal_probabilities_obs(
        self, X: pd.DataFrame, X_obs: pd.DataFrame
    ) -> np.ndarray:
        pass

    @abstractmethod
    def save_model(self, target_folder: str) -> None:
        """ Saves the model to files in the target folder.

        Args:
            target_file: Where to save the model parameters [folder name].

        Returns:
            Nothing, but stores model parameters in files in a folder at the
            path given.
        """
        pass
