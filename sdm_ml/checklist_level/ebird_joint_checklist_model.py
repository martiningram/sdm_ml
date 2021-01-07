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
        X_env: pd.DataFrame,
        X_checklist: pd.DataFrame,
        y_checklist: pd.DataFrame,
        checklist_cell_ids: np.ndarray,
    ):

        self.species_names = y_checklist.columns

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

    def predict_marginal_probabilities_direct():
        pass

    def predict_marginal_probabilities_obs():
        pass

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
