import os
import pickle
import numpy as np
from tqdm import tqdm
from svgp.tf.mogp_classifier import fit, predict_probs, predict_f_samples
from sdm_ml.presence_absence_model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler
from ml_tools.normals import calculate_log_joint_bernoulli_likelihood


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

    @classmethod
    def from_fit_result(cls, fit_result, scaler):

        base_object = cls(n_inducing=fit_result.mu.shape[1],
                          n_latent=fit_result.mu.shape[0],
                          kernel=fit_result.kernel)

        base_object.fit_result = fit_result
        base_object.scaler = scaler

        return base_object

    def fit(self, X, y):

        self.scaler = StandardScaler()

        X = self.scaler.fit_transform(X)

        self.fit_result = fit(
            X, y, self.n_inducing, self.n_latent, random_seed=self.seed,
            kernel=self.kernel)

    def predict_log_marginal_probabilities(self, X):

        X = self.scaler.transform(X)

        pred_probs = predict_probs(self.fit_result, X,
                                   n_draws=self.n_draws_predict)

        return np.stack([np.log(1 - pred_probs), np.log(pred_probs)], axis=-1)

    def calculate_log_likelihood(self, X, y):

        X = self.scaler.transform(X)

        site_iterator = predict_f_samples(self.fit_result, X,
                                          self.n_draws_predict)

        log_liks = np.zeros(X.shape[0])

        for i, cur_site_draws in enumerate(tqdm(site_iterator)):

            log_liks[i] = calculate_log_joint_bernoulli_likelihood(
                cur_site_draws, y[i])

        return log_liks

    def save_model(self, target_folder):

        os.makedirs(target_folder, exist_ok=True)

        # Save the results file
        np.savez(os.path.join(target_folder, 'results_file'),
                 **self.fit_result._asdict())

        # Save the scaler
        with open(os.path.join(target_folder, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
