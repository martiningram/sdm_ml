import os
import pickle
import numpy as np
from tqdm import tqdm
from svgp.tf.sogp_classifier import fit, predict_probs
from sdm_ml.presence_absence_model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler


class SOGP(PresenceAbsenceModel):

    def __init__(self, n_inducing, kernel, seed=2, n_draws_predict=1000):

        self.n_inducing = n_inducing
        self.kernel = kernel
        self.seed = seed
        self.n_draws_predict = n_draws_predict

        self.scaler = None
        self.fit_results = None

    @classmethod
    def from_fit_result(cls, fit_result, scaler):

        base_object = cls(n_inducing=fit_result.mu.shape[0],
                          kernel=fit_result.kernel)

        base_object.fit_result = fit_result
        base_object.scaler = scaler

        return base_object

    def fit(self, X, y):
        # TODO: Consider allowing the priors to change

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.fit_results = [
            fit(X, cur_y, self.n_inducing, random_seed=self.seed,
                kernel=self.kernel)
            for cur_y in tqdm(y.T)
        ]

    def predict_log_marginal_probabilities(self, X):

        X = self.scaler.transform(X)

        pred_probs = np.stack([
            predict_probs(cur_fit_result, X, n_draws=self.n_draws_predict)
            for cur_fit_result in self.fit_results
        ], axis=1)

        return np.stack([np.log(1 - pred_probs), np.log(pred_probs)], axis=-1)

    def calculate_log_likelihood(self, X, y):

        predictions = self.predict_log_marginal_probabilities(X)

        point_wise = y * predictions[..., 1] + (1 - y) * predictions[..., 0]

        return np.sum(point_wise, axis=1)

    def save_model(self, target_folder):

        os.makedirs(target_folder, exist_ok=True)

        # Save the results files
        for i, cur_results in enumerate(self.fit_results):
            np.savez(os.path.join(target_folder, f'results_file_{i}'),
                     **cur_results._asdict())

        # Save the scaler
        with open(os.path.join(target_folder, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
