import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
import pandas as pd


class ChecklistModel(ABC):
    @abstractmethod
    def fit(
        self,
        X_env: pd.DataFrame,
        X_checklist: pd.DataFrame,
        y_checklist: pd.DataFrame,
        checklist_cell_ids: np.ndarray,
    ) -> None:
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
