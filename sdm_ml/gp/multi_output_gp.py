import gpflow
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sdm_ml.model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler
from sdm_ml.gp.utils import find_starting_z, predict_with_link


class MultiOutputGP(PresenceAbsenceModel):

    def __init__(self, num_inducing=100, opt_steps=1000, n_draws_pred=4000,
                 verbose=False, rank=3):

        self.model = None
        self.scaler = None
        self.n_out = None

        self.num_inducing = num_inducing
        self.opt_steps = opt_steps
        self.verbose = verbose
        self.rank = rank
        self.n_draws_pred = n_draws_pred

    @staticmethod
    def prepare_stacked_features(X, n_out):

        n_total = X.shape[0]

        new_features = list()

        for cur_out in range(n_out):

            to_add = np.repeat(cur_out, n_total).reshape(-1, 1)

            cur_components = X
            cur_components = np.concatenate(
                [X, to_add], axis=1)

            new_features.append(cur_components)

        new_features = np.concatenate(new_features)

        return new_features

    @staticmethod
    def prepare_stacked_data(X, y):

        n_out = y.shape[1]

        new_features = MultiOutputGP.prepare_stacked_features(X, n_out)
        new_outcomes = np.reshape(y, (-1, 1), order='F')

        return new_features, new_outcomes

    def fit(self, X, y):

        self.n_out = y.shape[1]

        # Prepare kernel
        k1 = gpflow.kernels.RBF(X.shape[1], active_dims=range(X.shape[1]),
                                ARD=True)

        coreg = gpflow.kernels.Coregion(
            1, output_dim=self.n_out, rank=self.rank, active_dims=[X.shape[1]])

        kern = k1 * coreg
        lik = gpflow.likelihoods.Bernoulli()

        # Scale data
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Prepare data for kernel
        stacked_x, stacked_y = self.prepare_stacked_data(X, y)
        Z = find_starting_z(stacked_x, self.num_inducing)

        self.model = gpflow.models.SVGP(stacked_x, stacked_y.astype(np.float64),
                                        kern=kern, likelihood=lik, Z=Z)

        # Randomly initialise the coregionalisation weights to escape local
        # minimum described in:
        # https://gpflow.readthedocs.io/en/latest/notebooks/coreg_demo.html
        coreg = self.model.kern.children['kernels'][1]
        coreg.W = np.random.randn(self.n_out, self.rank)

        gpflow.train.ScipyOptimizer().minimize(
            self.model, maxiter=self.opt_steps, disp=self.verbose)

    def predict(self, X):

        assert(self.model is not None)

        X = self.scaler.transform(X)

        # Stack X to get multi-outputs
        x_stacked = self.prepare_stacked_features(X, self.n_out)

        # Predict this
        means, variances = self.model.predict_f(x_stacked)
        means, variances = np.squeeze(means), np.squeeze(variances)

        # Make draws
        draws = predict_with_link(means, variances, link_fun=norm.cdf,
                                  n_samples=self.n_draws_pred)

        pred_means = np.mean(draws, axis=0)

        # Reshape back using Fortran ordering
        pred_probs = pred_means.reshape((-1, self.n_out), order='F')

        return pred_probs

    def save_parameters(self, target_file):

        as_df = self.model.as_pandas_table()
        as_df.to_pickle(target_file)
