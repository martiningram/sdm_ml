import gpflow
import numpy as np
from sklearn.preprocessing import StandardScaler
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from sdm_ml.gp.utils import find_starting_z


class ICMMultiGP(MultiOutputGP):

    def __init__(self, rank=3, fixed_lengthscales=None, num_inducing=100,
                 opt_steps=500, n_draws_pred=4000, verbose=False):

        self.rank = rank
        self.fixed_lengthscales = fixed_lengthscales
        self.num_inducing = num_inducing

        super(ICMMultiGP, self).__init__(opt_steps=opt_steps,
                                         n_draws_pred=n_draws_pred,
                                         verbose=verbose)

    def fit(self, X, y):

        self.n_out = y.shape[1]

        kernel_args = dict()
        kernel_args['ARD'] = True

        if self.fixed_lengthscales is not None:
            kernel_args['lengthscales'] = self.fixed_lengthscales

        # Prepare kernel
        k1 = gpflow.kernels.RBF(X.shape[1], active_dims=range(X.shape[1]),
                                **kernel_args)

        if self.fixed_lengthscales is not None:
            k1.lengthscales.set_trainable(False)

        coreg = gpflow.kernels.Coregion(
            1, output_dim=self.n_out, rank=self.rank, active_dims=[X.shape[1]])

        kern = k1 * coreg
        lik = gpflow.likelihoods.Bernoulli()

        # Scale data
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Prepare data for kernel
        stacked_x, stacked_y = self.prepare_stacked_data(X, y)
        Z = find_starting_z(stacked_x, self.num_inducing)

        self.model = gpflow.models.SVGP(stacked_x, stacked_y.astype(np.float64),
                                        kern=kern, likelihood=lik, Z=Z)

        # Randomly initialise the coregionalisation weights to escape local
        # minimum described in:
        # https://gpflow.readthedocs.io/en/latest/notebooks/coreg_demo.html
        coreg = self.model.kern.children['kernels'][1]
        coreg.W = np.random.randn(self.n_out, self.rank)

        if self.verbose:
            print(self.model.as_pandas_table())
            print(k1.as_pandas_table()['trainable'])

        gpflow.train.ScipyOptimizer().minimize(
            self.model, maxiter=self.opt_steps, disp=self.verbose)

