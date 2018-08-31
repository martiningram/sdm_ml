import numpy as np
from tqdm import tqdm
from sdm_ml.model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV


class LogisticRegression(PresenceAbsenceModel):

    # TODO: Make this more general, so that any scikit learn model is accepted.
    # TODO: Add logging in a proper way.
    def __init__(self):

        self.models = list()
        self.scaler = None

    def fit(self, X, y):

        self.scaler = StandardScaler()

        # Transform features
        X = self.scaler.fit_transform(X)

        # We need to fit the marginals for this multi-species problem.
        for i in tqdm(range(y.shape[1])):
            cur_model = LogisticRegressionCV()
            cur_model.fit(X, y[:, i])
            self.models.append(cur_model)

    def predict(self, X):

        assert(len(self.models) > 0)
        predictions = list()

        X = self.scaler.transform(X)

        for cur_model in self.models:

            cur_predictions = cur_model.predict_proba(X)
            prob_present = cur_predictions[:, 1]
            predictions.append(prob_present)

        return np.stack(predictions, axis=1)
