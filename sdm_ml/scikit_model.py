import os
import pickle
import numpy as np
from tqdm import tqdm
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sdm_ml.presence_absence_model import PresenceAbsenceModel


class ScikitModel(PresenceAbsenceModel):

    def __init__(self, model_fun=partial(LogisticRegressionCV, cv=4,
                                         max_iter=int(1E4)),
                 clip_probs=True):

        # Clip probs avoids assigning probabilities of exactly 1 or 0 to
        # outcomes, in the same way as scikit learn does it for computing
        # the log loss.

        self.model_fun = model_fun
        self.models = list()
        self.scaler = None
        self.clip_probs = clip_probs

    def fit(self, X, y):

        self.scaler = StandardScaler()

        # Transform features
        X = self.scaler.fit_transform(X)

        # We need to fit the marginals for this multi-species problem.
        for i in tqdm(range(y.shape[1])):
            cur_model = self.model_fun()
            cur_model.fit(X, y[:, i])
            self.models.append(cur_model)

    @staticmethod
    def create_cross_validated_forest():

        search = GridSearchCV(RandomForestClassifier(), param_grid={
            'n_estimators': [50, 100, 250, 500, 1000],
            'max_depth': [None, 1, 2, 5],
            'max_features': [int(np.sqrt(8)), 2, 4]}, n_jobs=-1,
            cv=4, scoring="neg_log_loss")

        return search

    def predict_log_marginal_probabilities(self, X):

        assert(len(self.models) > 0)
        predictions = list()

        X = self.scaler.transform(X)

        for cur_model in self.models:

            cur_predictions = cur_model.predict_log_proba(X)
            predictions.append(cur_predictions)

        result = np.stack(predictions, axis=1)

        if self.clip_probs:
            result = np.clip(result, -15 * np.log(10), np.log(1 - 10**(-15)))

        return result

    def save_model(self, target_folder):
        # TODO: Might be better to save these individually rather than all
        # at once?

        os.makedirs(target_folder, exist_ok=True)

        # Pickle the model objects
        pickle.dump(self.models, open(
            os.path.join(target_folder, 'models.pkl'), 'bw'))

    def calculate_log_likelihood(self, X, y):
        # TODO: Make sure this is correct!

        # Predict marginal probabilities
        predictions = self.predict_log_marginal_probabilities(X)

        # Calculate log likelihood at each
        point_wise = y * predictions[..., 1] + (1 - y) * predictions[..., 0]

        # Sum across sites
        return np.sum(point_wise, axis=1)
