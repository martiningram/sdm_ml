import numpy as np
from svgp.tf.mogp_classifier import fit, predict_probs
from sdm_ml.presence_absence_model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler


class HierarchicalMOGP(PresenceAbsenceModel):

    def __init__(self, n_inducing, n_latent, kernel, seed=2,
                 n_draws_predict=1000):

        self.n_inducing = n_inducing
        self.n_latent = n_latent
        self.kernel = kernel
        self.seed = seed
        self.n_draws_predict = n_draws_predict

        self.scaler = None
        self.fit_result = None

    def fit(self, X, y):

        self.scaler = StandardScaler()

        X = self.scaler.fit_transform(X)

        self.fit_result = fit(
            X, y, self.n_inducing, self.n_latent, random_seed=self.random_seed,
            kernel=self.kernel)

    def predict_log_marginal_probabilities(self, X):

        X = self.scaler.transform(X)

        pred_probs = predict_probs(self.fit_result, X,
                                   n_draws=self.n_draws_predict)

        return np.log(pred_probs)
