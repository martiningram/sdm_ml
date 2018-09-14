import gpflow
import numpy as np
from sklearn.preprocessing import StandardScaler
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from sdm_ml.gp.utils import find_starting_z


class WartonGP(MultiOutputGP):

    def __init__(self, rank=16, num_inducing=100, opt_steps=500,
                 n_draws_pred=4000, verbose=False):

        self.rank = rank

        super(WartonGP, self).__init__(num_inducing=num_inducing,
                                       opt_steps=opt_steps,
                                       n_draws_pred=n_draws_pred,
                                       verbose=verbose)

    def fit(self, X, y):

        # TODO: This has to be set for the predict function to work, but it
        # would be nice to make that more obvious & enforce it somehow.
        self.n_out = y.shape[1]
        n_predictors = X.shape[1]

        # Get the kernel
        main_kern = gpflow.kernels.Linear(n_predictors, ARD=True,
                                          active_dims=range(n_predictors))
        white_kern = gpflow.kernels.White(1)
        coreg = gpflow.kernels.Coregion(
            1, output_dim=self.n_out, rank=self.rank,
            active_dims=[n_predictors])

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
