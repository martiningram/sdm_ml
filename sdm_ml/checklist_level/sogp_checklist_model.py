from .checklist_model import ChecklistModel
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from .likelihoods import compute_checklist_likelihood
from svgp.jax.helpers.sogp import (
    fit,
    ard_plus_bias_kernel_currier,
    gamma_default_lscale_prior_fn,
    constrain_positive,
)
from svgp.jax.helpers.svgp_spec import project_to_x
from ml_tools.normals import logistic_normal_integral_approx
import pickle
from typing import Callable
import os


class SOGPChecklistModel(ChecklistModel):
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

        def likelihood_fun(f, theta, m, log_duration, checklist_cell_ids, n_cells):

            env_logit = f
            obs_logit = log_duration * 0.42 - 3.1

            likelihood = compute_checklist_likelihood(
                env_logit, obs_logit, m, checklist_cell_ids, n_cells
            )

            return likelihood

        self.fit_results = list()
        self.species_names = species_names

        for cur_species in tqdm(species_names):

            cur_X_checklist = X_checklist(cur_species)
            n_c_obs = cur_X_checklist.shape[1]
            assert n_c_obs == 1, "Currently only log duration is supported!"
            cur_X_checklist = cur_X_checklist.reshape(-1)
            cur_m = 1 - y_checklist(cur_species)
            cur_cell_ids = checklist_cell_ids(cur_species)
            n_cells = np.max(cur_cell_ids) + 1

            init_kernel_params = {
                "log_lengthscales": np.log(np.random.uniform(2.0, 4.0, size=n_c_env)),
                "log_alpha": np.array(0.0),
                "log_bias_sd": np.array(0.0),
            }

            cur_lik_fun = lambda f, theta: likelihood_fun(
                f, theta, cur_m, cur_X_checklist, cur_cell_ids, n_cells
            )

            final_spec, final_theta = fit(
                X_env_cell,
                init_kernel_params,
                ard_plus_bias_kernel_currier,
                cur_lik_fun,
                gamma_default_lscale_prior_fn,
            )

            self.fit_results.append((final_spec, final_theta))

    def predict_log_marginal_probabilities(self, X: np.ndarray) -> np.ndarray:

        X = self.scaler.transform(X)

        predictions = list()

        for cur_species_name, cur_fit_result in zip(
            self.species_names, self.fit_results
        ):

            cur_mean, cur_var = project_to_x(cur_fit_result[0], X)

            prob_approx = logistic_normal_integral_approx(cur_mean, cur_var)

            cur_log_prob_abs = np.log(1 - prob_approx)
            cur_log_prob_pres = np.log(prob_approx)

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
                **cur_results[1],
            )

        # Save the scaler
        with open(os.path.join(target_folder, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
