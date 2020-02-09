import numpy as np
import gpflow as gpf
from gpflow.params import Parameter
from gpflow.decors import params_as_tensors


class MultiOutputMeanFunction(gpf.mean_functions.MeanFunction):

    def __init__(self, n_outputs):
        super(MultiOutputMeanFunction, self).__init__()

        c = np.zeros(n_outputs)
        c = np.reshape(c, (1, -1))

        self.c = Parameter(c)

    @params_as_tensors
    def __call__(self, X):

        return self.c
