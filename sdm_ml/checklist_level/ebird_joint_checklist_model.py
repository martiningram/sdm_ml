from .checklist_model import ChecklistModel
import numpy as np
from typing import Callable
import pandas as pd
from tqdm import tqdm
from .functional.hierarchical_checklist_model import fit
from .functional.hierarchical_checklist_model_mcmc import predict_env, predict_obs
from patsy import dmatrix
from jax_advi.advi import get_pickleable_subset
from ml_tools.utils import save_pickle_safely
from os import makedirs
from os.path import join
from sklearn.preprocessing import StandardScaler


class EBirdJointChecklistModel(ChecklistModel):
    def __init__(self, env_formula, obs_formula, M=20, n_draws=1000, verbose_fit=True):

        self.scaler = None
        self.M = M
        self.n_draws = n_draws
        self.verbose_fit = verbose_fit
        self.env_formula = env_formula
        self.obs_formula = obs_formula

    def fit(
        self,
        X_env: pd.DataFrame,
        X_checklist: pd.DataFrame,
        y_checklist: pd.DataFrame,
        checklist_cell_ids: np.ndarray,
    ):

        self.samples, self.advi_results, self.design_info = fit(
            X_env,
            X_checklist,
            y_checklist,
            checklist_cell_ids,
            self.env_formula,
            self.obs_formula,
            scale_env=False,
            draws=self.n_draws,
            M=self.M,
            verbose=self.verbose_fit,
        )

    def predict_marginal_probabilities_direct(self, X: pd.DataFrame) -> pd.DataFrame:

        return predict_env(X, self.samples, self.design_info)

    def predict_marginal_probabilities_obs(
        self, X: pd.DataFrame, X_obs: pd.DataFrame
    ) -> pd.DataFrame:

        return predict_obs(X, X_obs, self.samples, self.design_info)

    def save_model(self, target_folder: str) -> None:

        makedirs(target_folder, exist_ok=True)
        save_pickle_safely(
            {
                "env_coef_names": self.design_info["env"].column_names,
                "obs_coef_names": self.design_info["obs"].column_names,
                "env_scaler": None
                if "env_scaler" not in self.design_info
                else self.design_info["env_scaler"],
                "species_names": self.design_info["species_names"],
                "advi_results": get_pickleable_subset(self.advi_results),
                "env_formula": self.env_formula,
                "obs_formula": self.obs_formula,
            },
            join(target_folder, "design_info.pkl"),
        )
        self.samples.to_netcdf(join(target_folder, "draws.netcdf"))
