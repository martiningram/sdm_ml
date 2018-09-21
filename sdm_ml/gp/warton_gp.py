import gpflow
import numpy as np
from sklearn.preprocessing import StandardScaler
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from sdm_ml.gp.utils import find_starting_z
from dgplib.specialized_kernels import SwitchedKernel


class WartonGP(MultiOutputGP):

    def __init__(self, rank=16, inducing_per_class=3, opt_steps=500,
                 n_draws_pred=4000, verbose=False, minibatch_size=None,
                 use_rbf=False):

        # TODO: No need for opt_steps to be part of the parent class.

        self.rank = rank
        self.minibatch_size = minibatch_size
        self.inducing_per_class = inducing_per_class
        self.use_rbf = use_rbf

        super(WartonGP, self).__init__(opt_steps=opt_steps,
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

        # Find inducing points before stacking
        class_z = find_starting_z(X, self.inducing_per_class)
        Z = list()

        for cur_class in range(self.n_out):

            to_add = np.repeat(cur_class, self.inducing_per_class).reshape(
                -1, 1)
            cur_z = np.concatenate([class_z, to_add], axis=1)
            Z.append(cur_z)

        Z = np.concatenate(Z)

        # Prepare data for kernel
        stacked_x, stacked_y = self.prepare_stacked_data(X, y)

        self.model = gpflow.models.SVGP(
            stacked_x, stacked_y.astype(np.float64), kern=kern, likelihood=lik,
            Z=Z, minibatch_size=self.minibatch_size)

        if self.verbose:
            print(self.model.as_pandas_table())

        if self.minibatch_size is None:
            # Use L-BFGS
            gpflow.train.ScipyOptimizer().minimize(
                self.model, maxiter=self.opt_steps, disp=self.verbose)
        else:
            # Use Adam
            def callback(x):
                if x % 10 == 0 and self.verbose:
                    print(x, self.model.compute_log_likelihood() +
                          self.model.compute_log_prior())
            gpflow.train.AdamOptimizer().minimize(
                self.model, maxiter=self.opt_steps, step_callback=callback)
