from .checklist_model import ChecklistModel
import numpy as np
from typing import Callable
import pandas as pd
from tqdm import tqdm
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
from .functional.utils import predict_env_from_samples, predict_obs_from_samples
from ml_tools.patsy import remove_intercept_column


class EBirdJointChecklistModelStan(ChecklistModel):
    def __init__(self, model_file, env_formula, obs_formula):

        self.scaler = None
        self.stan_model = load_stan_model_cached(model_file)
        self.env_formula = env_formula
        self.obs_formula = obs_formula

    def fit(
        self,
        X_env: pd.DataFrame,
        X_checklist: pd.DataFrame,
        y_checklist: pd.DataFrame,
        checklist_cell_ids: np.ndarray,
    ):

        self.species_names = y_checklist.columns

        obs_design_mat = dmatrix(self.obs_formula, X_checklist)
        obs_covs = np.asarray(obs_design_mat)

        self.obs_design_info = obs_design_mat.design_info
        self.obs_cov_names = obs_design_mat.design_info.column_names

        # Create the design matrix for the environmental variables
        env_design_mat = dmatrix(self.env_formula, X_env)
        X_env = np.asarray(env_design_mat)

        X_env = remove_intercept_column(X_env, env_design_mat.design_info)

        self.env_cov_names = env_design_mat.design_info.column_names
        self.env_design_info = env_design_mat.design_info

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

    def predict_marginal_probabilities_direct(self, X: pd.DataFrame) -> pd.DataFrame:

        X_design = np.asarray(build_design_matrices([self.env_design_info], X)[0])
        X_design = remove_intercept_column(X_design, self.env_design_info)

        coef_draws = self.fit_results["env_slopes"]
        inter_draws = self.fit_results["env_intercepts"]
        probs = predict_env_from_samples(X_design, coef_draws, inter_draws)

        return pd.DataFrame(probs, columns=self.species_names)

    def predict_marginal_probabilities_obs(
        self, X: pd.DataFrame, X_obs: pd.DataFrame
    ) -> pd.DataFrame:

        env_covs = np.asarray(build_design_matrices([self.env_design_info], X)[0])
        obs_covs = np.asarray(build_design_matrices([self.obs_design_info], X_obs)[0])
        env_covs = remove_intercept_column(env_covs, self.env_design_info)

        coef_draws = self.fit_results["env_slopes"]
        inter_draws = self.fit_results["env_intercepts"]
        obs_coef_draws = self.fit_results["obs_coefs"]

        probs = predict_obs_from_samples(
            env_covs, coef_draws, inter_draws, obs_covs, obs_coef_draws
        )

        return pd.DataFrame(probs, columns=self.species_names)

    def save_model(self, target_folder: str) -> None:

        makedirs(target_folder, exist_ok=True)

        to_pickle = {
            "fit_result": self.fit_results.extract(),
            "obs_cov_names": self.obs_cov_names,
            "env_cov_names": self.env_cov_names,
            "species_names": self.species_names,
            "env_formula": self.env_formula,
            "obs_formula": self.obs_formula,
        }

        save_pickle_safely(to_pickle, join(target_folder, "fit_results.pkl"))
