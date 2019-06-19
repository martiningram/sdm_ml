import os
import gpflow
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from sdm_ml.model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler
from sdm_ml.gp.utils import find_starting_z, predict_and_summarise
from gpflow.multioutput import SeparateMixedMok, MixedKernelSharedMof


class NewStyleMultiGP(PresenceAbsenceModel):

    def __init__(self, num_inducing=100, opt_steps=500, verbose=False,
                 n_draws_pred=4000, L=8, use_variance_prior=True):

        self.num_inducing = num_inducing
        self.opt_steps = opt_steps
        self.verbose = verbose
        self.n_draws_pred = n_draws_pred
        self.L = L
        self.use_variance_prior = use_variance_prior

    def fit(self, X, y):

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        P = y.shape[1]
        n_pred = X.shape[1]

        with gpflow.defer_build():

            kern_list = []

            for _ in range(self.L - 1):

                cur_kern = gpflow.kernels.RBF(
                    n_pred, active_dims=range(n_pred), ARD=True)

                if self.use_variance_prior:
                    cur_kern.variance.prior = gpflow.priors.Gamma(2, 3)

                kern_list.append(cur_kern)

            # Add a bias kernel
            final_kern = gpflow.kernels.Bias(1)
            final_kern.variance.prior = gpflow.priors.Gamma(2, 3)
            kern_list.append(final_kern)

            kern = SeparateMixedMok(kern_list, W=np.random.randn(P, self.L))
            kern.W.prior = gpflow.priors.Gaussian(0, 1)

        lik = gpflow.likelihoods.Bernoulli()
        inducing = find_starting_z(X, self.num_inducing)

        M = inducing.shape[0]

        q_mu = np.zeros((M, self.L))
        q_sqrt = np.repeat(np.eye(M)[None, ...], self.L, axis=0) * 1.0

        features = MixedKernelSharedMof(
            gpflow.features.InducingPoints(inducing.copy()))

        self.model = gpflow.models.SVGP(X, y.astype(np.float32), kern, lik,
                                        features, q_mu=q_mu, q_sqrt=q_sqrt)

        gpflow.train.ScipyOptimizer().minimize(
            self.model, maxiter=self.opt_steps, disp=self.verbose)

    def predict(self, X):

        X = self.scaler.transform(X)

        means, vars = self.model.predict_f(X)
        means = np.squeeze(means)
        vars = np.squeeze(vars)

        # Reshape for the predict and summarise function
        reshaped_means = np.reshape(means, (-1,))
        reshaped_vars = np.reshape(vars, (-1,))

        summaries = predict_and_summarise(reshaped_means, reshaped_vars,
                                          n_samples=self.n_draws_pred)

        # Reshape back
        reshaped_probs = summaries.reshape(-1, means.shape[1])

        return reshaped_probs

    def save_parameters(self, target_folder):

        self.create_folder(target_folder)

        as_df = self.model.as_pandas_table()
        as_df.to_pickle(os.path.join(target_folder, 'new_multi_output_gp.pkl'))
