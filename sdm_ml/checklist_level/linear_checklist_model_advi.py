from .checklist_model import ChecklistModel
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax_advi.advi import (
    optimize_advi_mean_field,
    get_posterior_draws,
    get_pickleable_subset,
)
from .likelihoods import compute_checklist_likelihood
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from jax.nn import log_sigmoid
import os
from typing import Callable
import pickle
from jax.scipy.special import expit
import pandas as pd


class LinearChecklistModelADVI(ChecklistModel):
    def __init__(self):

        self.scaler = None
        self.fit_results = None
        self.species_names = None

    def fit(
        self,
        X_env_cell: pd.DataFrame,
        X_checklist: Callable[[str], pd.DataFrame],
        y_checklist: Callable[[str], np.ndarray],
        checklist_cell_ids: Callable[[str], np.ndarray],
        species_names: np.ndarray,
    ) -> None:

        self.scaler = StandardScaler()
        X_env_cell = self.scaler.fit_transform(X_env_cell)

        n_c_env = X_env_cell.shape[1]

        theta_shapes = {
            "env_coefs": (n_c_env),
            "env_intercept": (),
            "duration_slope": (),
            "duration_intercept": (),
        }

        def likelihood_fun(theta, m, X_env, duration, checklist_cell_ids, n_cells):

            env_logit = X_env @ theta["env_coefs"] + theta["env_intercept"]
            obs_logit = duration * theta["duration_slope"] + theta["duration_intercept"]

            likelihood = compute_checklist_likelihood(
                env_logit, obs_logit, m, checklist_cell_ids, n_cells
            )

            return jnp.sum(likelihood)

        prior_fun = lambda theta: jnp.sum(norm.logpdf(theta["env_coefs"]))

        self.fit_results = list()
        self.species_names = species_names

        for cur_species in tqdm(species_names):

            cur_X_checklist = np.log(
                X_checklist(cur_species)["duration_minutes"]
            ).values.reshape(-1, 1)

            n_c_obs = cur_X_checklist.shape[1]

            cur_X_checklist = cur_X_checklist.reshape(-1)

            cur_m = 1 - y_checklist(cur_species)
            cur_cell_ids = checklist_cell_ids(cur_species)
            n_cells = np.max(cur_cell_ids) + 1

            cur_lik_fun = partial(
                likelihood_fun,
                m=cur_m,
                X_env=X_env_cell,
                duration=cur_X_checklist,
                checklist_cell_ids=cur_cell_ids,
                n_cells=n_cells,
            )

            opt_result = optimize_advi_mean_field(
                theta_shapes, prior_fun, cur_lik_fun, n_draws=None
            )

            self.fit_results.append(opt_result)

    def predict_log_marginal_probabilities(self, X: np.ndarray) -> np.ndarray:

        X = self.scaler.transform(X)

        predictions = list()

        for cur_species_name, cur_fit_result in zip(
            self.species_names, self.fit_results
        ):

            def pred_fun(theta):

                cur_prediction = expit(X @ theta["env_coefs"] + theta["env_intercept"])

                return cur_prediction

            prob_pres = get_posterior_draws(
                cur_fit_result["free_means"],
                cur_fit_result["free_sds"],
                {},
                fun_to_apply=pred_fun,
            ).mean(axis=0)

            cur_log_prob_pres = jnp.log(prob_pres)
            cur_log_prob_abs = jnp.log(1 - prob_pres)

            predictions.append(np.stack([cur_log_prob_abs, cur_log_prob_pres], axis=1))

        predictions = np.stack(predictions, axis=1)

        return predictions

    def calculate_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        predictions = self.predict_log_marginal_probabilities(X)

        point_wise = y * predictions[..., 1] + (1 - y) * predictions[..., 0]

        return np.sum(point_wise, axis=1)

    def save_model(self, target_folder: str) -> None:

        os.makedirs(target_folder, exist_ok=True)

        # Save the results files
        for i, (cur_results, cur_species) in enumerate(
            zip(self.fit_results, self.species_names)
        ):
            with open(os.path.join(target_folder, f"results_file_{i}.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "species_name": cur_species,
                        "results": get_pickleable_subset(cur_results),
                    },
                    f,
                )

        # Save the scaler
        with open(os.path.join(target_folder, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
