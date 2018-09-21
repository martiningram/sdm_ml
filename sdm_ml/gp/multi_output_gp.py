import os
import gpflow
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from sdm_ml.model import PresenceAbsenceModel
from sdm_ml.gp.utils import predict_and_summarise


class MultiOutputGP(PresenceAbsenceModel):

    def __init__(self, opt_steps=1000, n_draws_pred=4000, verbose=False):

        self.model = None
        self.scaler = None
        self.n_out = None

        self.opt_steps = opt_steps
        self.verbose = verbose
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

    def predict(self, X):

        assert(self.model is not None)

        X = self.scaler.transform(X)

        # Stack X to get multi-outputs
        x_stacked = self.prepare_stacked_features(X, self.n_out)

        # Predict this
        means, variances = self.model.predict_f(x_stacked)
        means, variances = np.squeeze(means), np.squeeze(variances)

        pred_means = predict_and_summarise(
            means, variances, link_fun=norm.cdf, n_samples=self.n_draws_pred)

        # Reshape back using Fortran ordering
        pred_probs = pred_means.reshape((-1, self.n_out), order='F')

        return pred_probs

    def save_parameters(self, target_folder):

        self.create_folder(target_folder)

        as_df = self.model.as_pandas_table()
        as_df.to_pickle(os.path.join(target_folder, 'multi_output_gp.pkl'))
