import os
import numpy as np
import rpy2.robjects as robjects
from ml_tools.utils import get_cur_script_path
import rpy2.robjects.numpy2ri


# Maybe I can implement the Scikit pattern here?
# TODO: Consider actually inheriting. But I think for my purposes this is good
# enough.
class DismoBRT:

    def __init__(self, max_fit_attempts=1000):

        # Find R file.
        cur_script_path = get_cur_script_path(__file__)
        cur_folder = os.path.split(cur_script_path)[0]
        self.brt_file = os.path.join(cur_folder, 'brt_fit.R')
        self.max_fit_attempts = max_fit_attempts
        self.fit_attempts = 0

        assert os.path.isfile(self.brt_file)

        try:
            robjects.r['brtFit']
        except Exception:
            r_source = robjects.r['source']
            r_source(self.brt_file)

    def fit(self, X, y):

        # TODO: Is this the best place for this?
        rpy2.robjects.numpy2ri.activate()

        self.fit_fun = robjects.r['brtFit']

        r_type = 0
        self.fit_attempts = 0

        while r_type != 19 and self.fit_attempts < self.max_fit_attempts:

            self.n_tree, self.model = self.fit_fun(y, X)
            r_type = self.model.typeof
            self.fit_attempts += 1

    def predict_log_proba(self, X):

        rpy2.robjects.numpy2ri.activate()

        self.predict_fun = robjects.r['brtPredict']

        try:
            prob_pred = np.array(self.predict_fun(self.model, X, self.n_tree))
        except Exception:
            prob_pred = np.zeros(X.shape[0]) * np.nan

        # Add on the zero class to be Scikit Learn compatible
        neg_class = 1 - prob_pred

        result = np.stack([np.log(neg_class), np.log(prob_pred)], axis=1)

        return result
