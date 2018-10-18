import os
import pickle
import numpy as np
from tqdm import tqdm
from sdm_ml.model import PresenceAbsenceModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV


class ScikitModel(PresenceAbsenceModel):

    def __init__(self, model_fun=LogisticRegressionCV):

        self.model_fun = model_fun
        self.models = list()
        self.scaler = None

    def fit(self, X, y):

        self.scaler = StandardScaler()

        # Transform features
        X = self.scaler.fit_transform(X)

        # We need to fit the marginals for this multi-species problem.
        for i in tqdm(range(y.shape[1])):
            cur_model = self.model_fun()
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

    def save_parameters(self, target_folder):

        self.create_folder(target_folder)

        # Pickle the model objects
        pickle.dump(self.models, open(
            os.path.join(target_folder, 'params.pkl'), 'bw'))
