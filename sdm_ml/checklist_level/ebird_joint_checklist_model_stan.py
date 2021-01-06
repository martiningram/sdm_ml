from .checklist_model import ChecklistModel
import numpy as np
from typing import Callable
import pandas as pd
from tqdm import tqdm
from .functional.hierarchical_checklist_model import fit, predict_direct
from patsy import dmatrix, build_design_matrices
from jax_advi.advi import get_pickleable_subset
from ml_tools.utils import save_pickle_safely
from os import makedirs
from os.path import join
from sklearn.preprocessing import StandardScaler
from ml_tools.stan import load_stan_model_cached
from scipy.special import expit
from jax.nn import log_sigmoid
from sdm_ml.checklist_level.utils import evaluate_on_chunks


class EBirdJointChecklistModelStan(ChecklistModel):
    def __init__(self, model_file):

        self.scaler = None
        self.stan_model = load_stan_model_cached(model_file)

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
    def create_design_matrix_ebird_obs(X_checklist, design_info=None):

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

        if design_info is not None:
            obs_design_mat = build_design_matrices([design_info], obs_covariates)[0]
        else:
            obs_design_mat = dmatrix(
                "log_duration + protocol + time_of_day + protocol:log_duration",
                obs_covariates,
            )

        return obs_design_mat

    @staticmethod
    def create_design_matrix_env(X_env):

        # TODO: Maybe make interaction terms optional
        cov_names = X_env.columns

        main_effects = "+".join(cov_names)
        quad_terms = "+".join([f"I({x}**2)" for x in cov_names])
        inter_terms = "(" + main_effects + ")"
        inter_terms = inter_terms + ":" + inter_terms

        env_design_mat = dmatrix(
            main_effects + "+" + quad_terms + "+" + inter_terms + "- 1", X_env
        )

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

        self.obs_design_info = obs_design_mat.design_info

        self.obs_cov_names = obs_design_mat.design_info.column_names

        # Create the design matrix for the environmental variables
        env_design_mat = self.create_design_matrix_env(X_env)

        self.env_cov_names = env_design_mat.design_info.column_names

        self.scaler = StandardScaler()

        X_env = self.scaler.fit_transform(np.asarray(env_design_mat))

        model_data = {
            "K": X_checklist.shape[0],
            "N": X_env.shape[0],
            "cell_ids": checklist_cell_ids + 1,
            "n_obs_covs": obs_covs.shape[1],
            "n_env_covs": X_env.shape[1],
            "env_covs": X_env,
            "obs_covs": obs_covs,
            "n_species": y_checklist.shape[1],
            "y": y_checklist.values.astype(int),
        }

        self.fit_results = self.stan_model.sampling(data=model_data, thin=4)

    def predict_env_logits(self, X: pd.DataFrame) -> np.ndarray:

        X_design = self.create_design_matrix_env(X)
        X = self.scaler.transform(X_design)
        coef_draws = self.fit_results["env_slopes"]
        inter_draws = self.fit_results["env_intercepts"]

        # Do the prediction: (this might require too much RAM)
        logit_probs = np.einsum("nc,dcs->dns", X, coef_draws) + inter_draws.reshape(
            coef_draws.shape[0], 1, -1
        )

        return logit_probs

    def predict_marginal_probabilities_direct(self, X: pd.DataFrame) -> pd.DataFrame:
        def predict(X):
            logit_probs = self.predict_env_logits(X)
            prob = np.mean(expit(logit_probs), axis=0)
            return prob

        prob = evaluate_on_chunks(predict, 100, X, is_df=True)

        return pd.DataFrame(prob, index=X.index, columns=self.species_names)

    def predict_marginal_probabilities_obs(
        self, X: pd.DataFrame, X_obs: pd.DataFrame
    ) -> pd.DataFrame:
        def predict(X, X_obs):

            env_logit_probs = self.predict_env_logits(X)
            obs_coef_draws = self.fit_results["obs_coefs"]

            obs_design_mat = self.create_design_matrix_ebird_obs(
                X_obs, self.obs_design_info
            )

            obs_design_array = np.asarray(obs_design_mat)
            obs_logits = np.einsum("nc,dcs->dns", obs_design_array, obs_coef_draws)

            probs = np.exp(log_sigmoid(env_logit_probs) + log_sigmoid(obs_logits))
            return np.mean(probs, axis=0)

        result = evaluate_on_chunks(predict, 100, X, X_obs, is_df=True)

        return pd.DataFrame(result, columns=self.species_names)

    def save_model(self, target_folder: str) -> None:

        makedirs(target_folder, exist_ok=True)

        to_pickle = {
            "fit_result": self.fit_results.extract(),
            "obs_cov_names": self.obs_cov_names,
            "env_cov_names": self.env_cov_names,
            "species_names": self.species_names,
        }

        save_pickle_safely(to_pickle, join(target_folder, "fit_results.pkl"))
        save_pickle_safely(self.scaler, join(target_folder, "scaler.pkl"))
