from .checklist_model import ChecklistModel
import numpy as np
import pandas as pd
import jax.numpy as jnp
from .likelihoods import compute_checklist_likelihood
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from jax.nn import log_sigmoid
import os
from typing import Callable
import pickle
from .functional.max_lik_occu_model import fit, predict_env_logit
from ml_tools.patsy import create_formula


class LinearChecklistModel(ChecklistModel):
    def __init__(self, env_formula, det_formula):

        self.fit_results = None
        self.species_names = None
        self.env_formula = env_formula
        self.det_formula = det_formula

    def fit(
        self,
        X_env_cell: pd.DataFrame,
        X_checklist: Callable[[str], pd.DataFrame],
        y_checklist: Callable[[str], np.ndarray],
        checklist_cell_ids: Callable[[str], np.ndarray],
        species_names: np.ndarray,
    ) -> None:

        self.fit_results = list()
        self.species_names = species_names

        for cur_species in tqdm(species_names):

            cur_X_checklist = X_checklist(cur_species)
            cur_y_checklist = y_checklist(cur_species)
            cur_cell_ids = checklist_cell_ids(cur_species)

            fit_result = fit(
                X_env_cell,
                cur_X_checklist,
                cur_y_checklist,
                cur_cell_ids,
                self.env_formula,
                self.det_formula,
                scale_env_data=True,
            )

            self.fit_results.append(fit_result)

    def predict_log_marginal_probabilities(self, X: pd.DataFrame) -> np.ndarray:

        predictions = list()

        for cur_species_name, cur_fit_result in zip(
            self.species_names, self.fit_results
        ):

            cur_prediction = predict_env_logit(
                X,
                self.env_formula,
                cur_fit_result["env_coefs"],
                cur_fit_result["env_scaler"],
            )

            cur_log_prob_pres = log_sigmoid(cur_prediction)
            cur_log_prob_abs = log_sigmoid(-cur_prediction)

            predictions.append(np.stack([cur_log_prob_abs, cur_log_prob_pres], axis=1))

        predictions = np.stack(predictions, axis=1)

        return predictions

    def calculate_log_likelihood(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:

        predictions = self.predict_log_marginal_probabilities(X)

        point_wise = y * predictions[..., 1] + (1 - y) * predictions[..., 0]

        return np.sum(point_wise, axis=1)

    def save_model(self, target_folder: str) -> None:

        os.makedirs(target_folder, exist_ok=True)

        # Save the results files
        for i, (cur_results, cur_species) in enumerate(
            zip(self.fit_results, self.species_names)
        ):
            np.savez(
                os.path.join(target_folder, f"results_file_{i}"),
                species_name=cur_species,
                env_coefs=cur_results["env_coefs"],
                obs_coefs=cur_results["obs_coefs"],
                successful=cur_results["optimisation_successful"],
            )

        # Save the scaler
        with open(os.path.join(target_folder, "scaler.pkl"), "wb") as f:
            pickle.dump(self.fit_results[0]["env_scaler"], f)
