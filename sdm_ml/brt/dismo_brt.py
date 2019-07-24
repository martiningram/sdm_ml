import os
import numpy as np
import rpy2.robjects as robjects
from python_tools.paths import get_cur_script_path
import rpy2.robjects.numpy2ri


# Maybe I can implement the Scikit pattern here?
# TODO: Consider actually inheriting. But I think for my purposes this is good
# enough.
class DismoBRT:

    def __init__(self):

        # Find R file.
        cur_script_path = get_cur_script_path(__file__)
        cur_folder = os.path.split(cur_script_path)[0]
        brt_file = os.path.join(cur_folder, 'brt_fit.R')

        assert os.path.isfile(brt_file)

        r_source = robjects.r['source']
        r_source("./brt_fit.R")

        self.fit_fun = robjects.r['brtFit']
        self.predict_fun = robjects.r['brtPredict']

    def fit(self, X, y):

        # TODO: Is this the best place for this?
        rpy2.robjects.numpy2ri.activate()

        self.n_tree, self.model = self.fit_fun(y, X)

    def predict_proba(self, X):

        rpy2.robjects.numpy2ri.activate()

        prob_pred = np.array(self.predict_fun(self.model, X, self.n_tree))

        # Add on the zero class to be Scikit Learn compatible
        neg_class = 1 - prob_pred

        result = np.stack([neg_class, prob_pred], axis=1)

        return result
