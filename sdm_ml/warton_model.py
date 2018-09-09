from __future__ import print_function

import os
import numpy as np
from scipy.special import expit
from sdm_ml.model import PresenceAbsenceModel
from python_tools.paths import get_cur_script_path
from python_tools.utils import load_stan_model_cached


class WartonModel(PresenceAbsenceModel):

    def __init__(self, n_latents=4):

        cur_script_path = get_cur_script_path(__file__)
        script_dir = os.path.split(cur_script_path)[0]

        stan_path = os.path.join(script_dir, 'warton_latent_bernoulli.stan')
        assert(os.path.isfile(stan_path))

        self.stan_model = load_stan_model_cached(stan_path)
        self.n_latents = n_latents

    def fit(self, X, y):

        # Prepare stan dict
        stan_dict = {
            'n': X.shape[0],
            'k': y.shape[1],
            'p': X.shape[1],
            'l': self.n_latents,
            'y': y.astype(int),
            'X': X
        }

        self.fit = self.stan_model.sampling(data=stan_dict)

    def predict(self, X):

        # X is [N, P].
        # beta_1 should be of shape [S, P, K], where P is the number of
        # features, and K is the number of species, and S is the number of
        # posterior samples.
        beta_1 = self.fit.extract('beta_1')['beta_1']
        site_means = list()

        # Get the logits at each site -- this will be [S, K].
        for cur_site in X:
            # Calculate the samples here
            # 1 x P
            cur_covariates = np.expand_dims(cur_site, axis=0)

            # 1 x P times [S, P, K] gives [S, 1, K]
            cur_logits = np.matmul(cur_covariates, beta_1)
            cur_probs = expit(cur_logits)

            # Calculate the mean along axis 0 and squeeze
            cur_mean = np.squeeze(np.mean(cur_probs, axis=0))
            site_means.append(cur_mean)

        site_means = np.stack(site_means, axis=0)

        # TODO: I may have to add on the variance implied by the latent model
        # here.
        return site_means

    def save_parameters(self, target_folder):

        self.create_folder(target_folder)

        # Store the model samples
        samples = self.fit.extract()
        np.savez(os.path.join(target_folder, 'stan_draws'), **samples)

        # Store the output of the print method
        print_file = os.path.join(target_folder, 'fit_results.txt')
        with open(print_file, 'w') as f:
            print(self.fit, file=f)

        # Store the samples of the implied covariance matrix
        cov_samples = np.matmul(
            samples['beta_z'], samples['beta_z'].transpose(0, 2, 1))

        np.save(os.path.join(target_folder, 'cov_mat'), cov_samples)
