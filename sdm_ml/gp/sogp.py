import os
import pickle
import numpy as np
from tqdm import tqdm
from svgp.jax.helpers.sogp import (
    fit_bernoulli_sogp,
    ard_plus_bias_kernel_currier,
    gamma_default_lscale_prior_fn,
)
from svgp.jax.helpers.svgp_spec import project_to_x
from sdm_ml.presence_absence_model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler
from ml_tools.jax_kernels import ard_rbf_kernel, matern_kernel_32
from functools import partial
from ml_tools.normals import normal_cdf_integral


class SOGP(PresenceAbsenceModel):
    def __init__(self, n_inducing, kernel, seed=2):

        assert kernel in [
            "matern_32",
            "rbf",
        ], "Only matern_32 and rbf kernels supported for now!"

        self.base_kernel_fn = (
            matern_kernel_32 if kernel == "matern_32" else ard_rbf_kernel
        )

        self.n_inducing = n_inducing
        self.seed = seed

        self.scaler = None
        self.fit_results = None

    @classmethod
    def from_fit_result(cls, fit_result, scaler):

        base_object = cls(n_inducing=fit_result.mu.shape[0], kernel=fit_result.kernel)

        base_object.fit_result = fit_result
        base_object.scaler = scaler

        return base_object

    def fit(self, X, y):
        # TODO: Consider allowing the priors to change

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        init_params = {
            "log_lengthscales": np.random.uniform(2.0, 4.0, size=X.shape[1]),
            "log_alpha": np.array(0.0),
            "log_bias_sd": np.array(0.0),
        }

        self.fit_results = [
            fit_bernoulli_sogp(
                X=X,
                y=cur_y,
                init_params=init_params,
                kernel_currier=partial(
                    ard_plus_bias_kernel_currier, base_kernel=self.base_kernel_fn
                ),
                prior_fun=gamma_default_lscale_prior_fn,
                n_inducing=self.n_inducing,
                random_seed=self.seed,
            )
            for cur_y in tqdm(y.T)
        ]

    def predict_log_marginal_probabilities(self, X):

        X = self.scaler.transform(X)

        pred_means_and_vars = [
            project_to_x(cur_fit_result[0], X) for cur_fit_result in self.fit_results
        ]

        pred_probs = np.stack(
            [normal_cdf_integral(x[0], x[1]) for x in pred_means_and_vars], axis=1
        )

        return np.stack([np.log(1 - pred_probs), np.log(pred_probs)], axis=-1)

    def calculate_log_likelihood(self, X, y):

        predictions = self.predict_log_marginal_probabilities(X)

        point_wise = y * predictions[..., 1] + (1 - y) * predictions[..., 0]

        return np.sum(point_wise, axis=1)

    def save_model(self, target_folder):

        os.makedirs(target_folder, exist_ok=True)

        # Save the results files
        for i, cur_results in enumerate(self.fit_results):
            np.savez(
                os.path.join(target_folder, f"results_file_{i}"), **cur_results[1],
            )

        # Save the scaler
        with open(os.path.join(target_folder, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
