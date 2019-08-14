import numpy as np


# This is the simplest model I could think of. For use with a ScikitModel.
class BaseRateModel:

    def __init__(self):

        self.base_rate = None

    def fit(self, X, y):

        self.base_rate = y.mean()

    def predict_log_proba(self, X):

        assert self.base_rate is not None, \
            "Model must be fit before predicting!"

        log_p = np.log(self.base_rate)
        log_q = np.log(1 - self.base_rate)

        one_probs = np.ones(X.shape[0]) * log_p
        zero_probs = np.ones(X.shape[0]) * log_q

        return np.stack([zero_probs, one_probs], axis=1)
