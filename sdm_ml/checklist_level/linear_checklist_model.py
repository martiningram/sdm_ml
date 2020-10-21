from .checklist_model import ChecklistModel
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax.numpy as jnp
from ml_tools.constrain import apply_transformation
from jax.scipy.stats import norm
from ml_tools.max_lik import find_map_estimate
from .likelihoods import compute_checklist_likelihood
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from jax.nn import log_sigmoid
import os
from typing import Callable
import pickle


class LinearChecklistModel(ChecklistModel):
    def __init__(self):

        self.scaler = None
        self.fit_results = None
        self.species_names = None

    def fit(
        self,
        X_env_cell: np.ndarray,
        X_checklist: Callable[[str], np.ndarray],
        y_checklist: Callable[[str], np.ndarray],
        checklist_cell_ids: Callable[[str], np.ndarray],
        species_names: np.ndarray,
    ) -> None:

        self.scaler = StandardScaler()
        X_env_cell = self.scaler.fit_transform(X_env_cell)

        n_c_env = X_env_cell.shape[1]

        def likelihood_fun(theta, m, X_env, X_checklist, checklist_cell_ids, n_cells):

            env_logit = X_env @ theta["env_coefs"] + theta["env_intercept"]
            # obs_logit = X_checklist @ theta["obs_coefs"]
            obs_logit = X_checklist * 0.42 - 3.1

            likelihood = compute_checklist_likelihood(
                env_logit, obs_logit, m, checklist_cell_ids, n_cells
            )

            return jnp.sum(likelihood)

        self.fit_results = list()
        self.species_names = species_names

        for cur_species in tqdm(species_names):

            cur_X_checklist = X_checklist(cur_species)

            n_c_obs = cur_X_checklist.shape[1]

            assert n_c_obs == 1, "Currently only log duration is supported!"

            cur_X_checklist = cur_X_checklist.reshape(-1)

            init_theta = {
                "env_coefs": np.zeros(n_c_env),
                "env_intercept": np.array(0.0),
            }

            cur_m = 1 - y_checklist(cur_species)
            cur_cell_ids = checklist_cell_ids(cur_species)
            n_cells = np.max(cur_cell_ids) + 1

            cur_lik_fun = partial(
                likelihood_fun,
                m=cur_m,
                X_env=X_env_cell,
                X_checklist=cur_X_checklist,
                checklist_cell_ids=cur_cell_ids,
                n_cells=n_cells,
            )

            final_theta, opt_result = find_map_estimate(
                init_theta,
                cur_lik_fun,
                prior_fun=lambda theta: jnp.sum(norm.logpdf(theta["env_coefs"]))
                # + jnp.sum(norm.logpdf(theta["obs_coefs"])),
            )

            self.fit_results.append(final_theta)

    def predict_log_marginal_probabilities(self, X: np.ndarray) -> np.ndarray:

        X = self.scaler.transform(X)

        predictions = list()

        for cur_species_name, cur_fit_result in zip(
            self.species_names, self.fit_results
        ):

            cur_prediction = (
                X @ cur_fit_result["env_coefs"] + cur_fit_result["env_intercept"]
            )

            cur_log_prob_pres = log_sigmoid(cur_prediction)
            cur_log_prob_abs = log_sigmoid(-cur_prediction)

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
            np.savez(
                os.path.join(target_folder, f"results_file_{i}"),
                species_name=cur_species,
                **cur_results,
            )

        # Save the scaler
        with open(os.path.join(target_folder, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
