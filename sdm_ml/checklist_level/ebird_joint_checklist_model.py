from .checklist_model import ChecklistModel
import numpy as np
from typing import Callable
import pandas as pd
from tqdm import tqdm
from .functional.hierarchical_checklist_model import fit, predict_direct
from patsy import dmatrix
from jax_advi.advi import get_pickleable_subset
from ml_tools.utils import save_pickle_safely
from os import makedirs
from os.path import join
from sklearn.preprocessing import StandardScaler


class EBirdJointChecklistModel(ChecklistModel):
    def __init__(
        self, M=20, n_pred_draws=1000, verbose_fit=True, env_interactions=True
    ):

        self.scaler = None
        self.M = M
        self.n_pred_draws = n_pred_draws
        self.verbose_fit = verbose_fit
        self.env_interactions = env_interactions

    @staticmethod
    def assemble_dataset(
        X_env_cell, X_checklist_fun, y_checklist, checklist_cell_ids, species_names
    ):

        # Assemble the dataset
        y_full = list()
        cell_ids = None
        X_checklist = None

        print("Preparing data...")
        for cur_species in tqdm(species_names):

            cur_y = y_checklist(cur_species)
            cur_cell_ids = checklist_cell_ids(cur_species)
            y_full.append(cur_y)

            if cell_ids is None:
                cell_ids = cur_cell_ids
            else:
                assert np.all(cur_cell_ids == cell_ids)

            cur_X_checklist = X_checklist_fun(cur_species)

            if X_checklist is None:
                X_checklist = cur_X_checklist
            else:
                assert np.all(
                    X_checklist["sampling_event_identifier"].values
                    == cur_X_checklist["sampling_event_identifier"].values
                )

        y_full = np.stack(y_full, axis=1)

        # TODO: Is that right?
        X_env = X_env_cell
        print("Done.")

        return X_checklist, y_full, X_env, cell_ids

    @staticmethod
    def create_design_matrix_ebird_obs(X_checklist):

        protocol_types = (
            X_checklist["protocol_type"]
            .str.replace("Traveling - Property Specific", "Traveling")
            .values
        )

        hour_of_day = (
            X_checklist["time_observations_started"]
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

        log_duration = np.log(X_checklist["duration_minutes"].values)

        obs_covariates = {
            "time_of_day": time_of_day,
            "protocol": protocol_types,
            "log_duration": log_duration,
        }

        obs_design_mat = dmatrix(
            "log_duration + protocol + time_of_day + protocol:log_duration",
            obs_covariates,
        )

        return obs_design_mat

    @staticmethod
    def create_design_matrix_env(X_env, add_interactions=True):

        cov_names = X_env.columns

        main_effects = "+".join(cov_names)
        quad_terms = "+".join([f"I({x}**2)" for x in cov_names])

        model_str = main_effects + "+" + quad_terms

        if add_interactions:
            inter_terms = "(" + main_effects + ")"
            inter_terms = inter_terms + ":" + inter_terms
            model_str = model_str + "+" + inter_terms

        env_design_mat = dmatrix(model_str + "- 1", X_env)

        return env_design_mat

    def fit(
        self,
        X_env_cell: pd.DataFrame,
        X_checklist: Callable[[str], pd.DataFrame],
        y_checklist: Callable[[str], np.ndarray],
        checklist_cell_ids: Callable[[str], np.ndarray],
        species_names: np.ndarray,
    ):

        self.species_names = species_names

        X_checklist, y_full, X_env, cell_ids = self.assemble_dataset(
            X_env_cell, X_checklist, y_checklist, checklist_cell_ids, species_names
        )

        # Next, extract the checklist data we need
        obs_design_mat = self.create_design_matrix_ebird_obs(X_checklist)
        obs_covs = np.asarray(obs_design_mat)

        self.obs_cov_names = obs_design_mat.design_info.column_names

        # Create the design matrix for the environmental variables
        env_design_mat = self.create_design_matrix_env(X_env, self.env_interactions)

        self.env_cov_names = env_design_mat.design_info.column_names

        self.scaler = StandardScaler()

        X_env = self.scaler.fit_transform(np.asarray(env_design_mat))

        self.fit_result = fit(
            X_env, obs_covs, y_full, cell_ids, M=self.M, verbose=self.verbose_fit
        )

    def predict_log_marginal_probabilities(self, X: pd.DataFrame) -> np.ndarray:

        X_design = self.create_design_matrix_env(X, self.env_interactions)
        X = self.scaler.transform(X_design)

        preds = predict_direct(self.fit_result, X, n_draws=self.n_pred_draws)

        log_preds = np.log(preds)
        log_not_preds = np.log(1 - preds)

        combined = np.stack([log_not_preds, log_preds], axis=-1)

        return combined

    def calculate_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        predictions = self.predict_log_marginal_probabilities(X)

        point_wise = y * predictions[..., 1] + (1 - y) * predictions[..., 0]

        return np.sum(point_wise, axis=1)

    def save_model(self, target_folder: str) -> None:

        makedirs(target_folder, exist_ok=True)

        to_pickle = {
            "fit_result": get_pickleable_subset(self.fit_result),
            "obs_cov_names": self.obs_cov_names,
            "env_cov_names": self.env_cov_names,
            "species_names": self.species_names,
        }

        save_pickle_safely(to_pickle, join(target_folder, "fit_results.pkl"))
        save_pickle_safely(self.scaler, join(target_folder, "scaler.pkl"))
