import os
import gpflow
from os.path import join
from scipy.stats import norm
from scipy.special import logsumexp
from .utils import (find_starting_z, calculate_log_joint_bernoulli_likelihood,
                    save_gpflow_model)


class SingleOutputGP(PresenceAbsenceModel):

    def __init__(self, n_inducing, kernel_function, maxiter=int(1E6),
                 verbose_fit=True, n_draws_predict=int(1E4)):

        self.models = None
        self.is_fit = False
        self.kernel_function = kernel_function
        self.n_inducing = n_inducing
        self.maxiter = maxiter
        self.verbose_fit = verbose_fit
        self.n_draws_predict = n_draws_predict

    def fit(self, X, y):

        Z = find_starting_z(X, num_inducing=n_inducing, use_minibatching=False)

        self.models = list()

        # We need to fit each species separately
        for cur_output in range(y.shape[1]):

            cur_kernel = self.kernel_function()
            cur_likelihood = gpflow.likelihoods.Bernoulli()

            cur_y = y[:, [cur_output]]

            cur_m = gpflow.models.SVGP(X, cur_y, kern=cur_kernel,
                                       likelihood=cur_likelihood, Z=Z)

            opt = gpflow.train.ScipyOptimizer(
                options={'maxfun': self.maxiter})

            opt.minimize(cur_m, maxiter=self.maxiter, disp=self.verbose_fit)

            self.models.append(cur_m)

    def predict_log_marginal_probabilities(self, X: np.ndarray) -> np.ndarray:
        # TODO: Check against GPFlow.
        # TODO: Is this really worth it? Could just use predict_y.

        # Run the prediction for each model
        results = list()

        for cur_model in self.models:

           # Predict f, the latent probability on the probit scale
           f_mean, f_var = cur_model.predict_f(X)
           f_std = np.sqrt(f_var)

           # TODO:
           # This is a bit of a balance. Predict all at once, or not?
           # I think here, all at once is OK. But it could blow up if the
           # number of examples is large.
           # Could use numba or something to do it on a per-site basis,
           # or chunk.
           draws = np.random.normal(
               f_mean, f_std, size=(self.n_draws_predict, f_mean.shape[0]))

           # OK, now to do the logsumexp trick.
           pre_factor = -np.log(self.n_draws_predict)
           draw_log_probs = norm.logcdf(draws)

           result = logsumexp(pre_factor + draw_log_probs, axis=0)

           results.append(result)

        results = np.stack(results, axis=1)

        return results

    def calculate_log_likelihood(self, X, y):

        assert y.shape[1] == len(self.models)

        means, sds = list(), list()

        for cur_model in self.models:

            cur_mean, cur_vars = cur_model.predict_f(X)
            cur_sds = np.sqrt(cur_vars)

            means.append(cur_mean)
            sds.append(cur_sds)

        means = np.stack(means, axis=1)
        sds = np.stack(sds, axis=1)

        site_log_liks = np.zeros(means.shape[0])

        # Estimate site by site
        for i, (cur_y, cur_mean, cur_sd) in enumerate(zip(y, means, sds)):

            draws = np.random.normal(
                cur_mean, cur_sd, size=(self.n_draws_predict, means.shape[1]))

            log_lik = calculate_log_joint_bernoulli_likelihood(draws, cur_y)

            site_log_liks[i] = log_lik

        return site_log_liks

    def save_model(self, target_folder):

        os.makedirs(target_folder, exist_ok=True)

        for i, cur_model in enumerate(self.models):

            target_file = join(target_folder, f'model_species_{i}')

            save_gpflow_model(cur_model, target_file)
