import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, average_precision_score


class Evaluator(object):

    def __init__(self, dataset):

        self.dataset = dataset

    # @staticmethod
    # def log_loss(y_t, y_p):
    #     # Calculates the negative mean log likelihood

    #     pre_log = y_p.copy()
    #     did_not_happen = y_t == 0
    #     pre_log[did_not_happen] = 1 - pre_log[did_not_happen]

    #     log_likelihoods = np.log(pre_log)
    #     mean = np.mean(log_likelihoods)

    #     return -mean

    def evaluate_model(self, model):

        training_set = self.dataset.get_training_set()
        X, y = training_set['covariates'], training_set['outcomes']

        # Fit the model
        model.fit(X.values, y.values)

        # Get the test set
        test_set = self.dataset.get_test_set()
        X_t, y_t = test_set['covariates'], test_set['outcomes']

        # Predict
        y_p = model.predict(X_t.values)
        y_p = pd.DataFrame(y_p, columns=y_t.columns, index=y_t.index)

        results = list()

        for cur_species in y_p.columns:

            predictions = y_p[cur_species].values
            truth = y_t[cur_species].values.astype(int)

            lloss = log_loss(truth, predictions, labels=np.array([0, 1]))

            # TODO: Make sure I got that right!
            deviance = lloss * 2 * predictions.shape[0]

            cur_metrics = {
                'log_loss': log_loss(truth, predictions,
                                     labels=np.array([0, 1])),
                'species': cur_species,
                'deviance': deviance,
                'average_precision': average_precision_score(
                    truth, predictions)
            }

            results.append(cur_metrics)

        return pd.DataFrame(results)
