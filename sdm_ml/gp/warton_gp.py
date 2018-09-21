import gpflow
import numpy as np
from sklearn.preprocessing import StandardScaler
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from sdm_ml.gp.utils import find_starting_z
from dgplib.specialized_kernels import SwitchedKernel


class WartonGP(MultiOutputGP):

    def __init__(self, rank=16, num_inducing=100, opt_steps=500,
                 n_draws_pred=4000, verbose=False, use_rbf=False):

        self.rank = rank
        self.use_rbf = use_rbf

        super(WartonGP, self).__init__(num_inducing=num_inducing,
                                       opt_steps=opt_steps,
                                       n_draws_pred=n_draws_pred,
                                       verbose=verbose)

    def fit(self, X, y):

        # TODO: This has to be set for the predict function to work, but it
        # would be nice to make that more obvious & enforce it somehow.
        self.n_out = y.shape[1]
        n_predictors = X.shape[1]

        # Kernel args are the same for RBF and linear kernels:
        kernel_args = dict(input_dim=n_predictors,
                           ARD=True,
                           active_dims=range(n_predictors))

        if self.use_rbf:
            # Use the RBF kernel
            kernels = [gpflow.kernels.RBF(**kernel_args) +
                gpflow.kernels.Bias(1) for _ in range(self.n_out)]
        else:
            # Use a linear kernel
            kernels = [gpflow.kernels.Linear(**kernel_args) +
                gpflow.kernels.Bias(1) for _ in range(self.n_out)]

        main_kern = SwitchedKernel(kernels, self.n_out)

        with gpflow.defer_build():
            white_kern = gpflow.kernels.White(1)
            coreg = gpflow.kernels.Coregion(
                1, output_dim=self.n_out, rank=self.rank,
                active_dims=[n_predictors])
            coreg.W = np.random.randn(self.n_out, self.rank)
            white_kern.variance = 1
            white_kern.variance.set_trainable(False)

        kern = main_kern + (coreg * white_kern)

        # TODO: Lots of copying here from the ICM model. Fix.
        lik = gpflow.likelihoods.Bernoulli()

        # Scale data
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Prepare data for kernel
        stacked_x, stacked_y = self.prepare_stacked_data(X, y)
        Z = find_starting_z(stacked_x, self.num_inducing)

        self.model = gpflow.models.SVGP(stacked_x, stacked_y.astype(np.float64),
                                        kern=kern, likelihood=lik, Z=Z)

        if self.verbose:
            print(self.model.as_pandas_table())

        gpflow.train.ScipyOptimizer().minimize(
            self.model, maxiter=self.opt_steps, disp=self.verbose)
