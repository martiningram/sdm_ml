import gpflow
import numpy as np
from sklearn.preprocessing import StandardScaler
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from sdm_ml.gp.utils import find_starting_z


class LMCMultiGP(MultiOutputGP):

    def __init__(self, ranks=[8, 8], num_inducing=100, opt_steps=500,
                 n_draws_pred=4000, verbose=False):

        self.ranks = ranks
        self.num_inducing = num_inducing

        super(LMCMultiGP, self).__init__(opt_steps=opt_steps,
                                         n_draws_pred=n_draws_pred,
                                         verbose=verbose)

    @staticmethod
    def produce_kernel(n_predictors, ranks, n_out):

        # TODO: Maybe make this more flexible.
        kernels = list()

        with gpflow.defer_build():

            for cur_rank in ranks:

                cur_k = gpflow.kernels.RBF(n_predictors,
                                           active_dims=range(n_predictors),
                                           ARD=True)

                cur_k.variance.prior = gpflow.priors.Gamma(2, 3)

                cur_coreg = gpflow.kernels.Coregion(
                    1, output_dim=n_out, rank=cur_rank,
                    active_dims=[n_predictors])

                cur_coreg.W.prior = gpflow.priors.Gaussian(0, 1)

                # Also initialise the coregionalisation weights randomly
                cur_coreg.W = np.random.randn(n_out, cur_rank)

                cur_kern = cur_k * cur_coreg

                kernels.append(cur_kern)

        # Sum the kernels together
        total_k = kernels[0]

        if len(kernels) > 1:
            for other_kern in kernels[1:]:
                total_k = total_k + other_kern

        return total_k

    def fit(self, X, y):

        # TODO: This has to be set for the predict function to work, but it
        # would be nice to make that more obvious & enforce it somehow.
        self.n_out = y.shape[1]
        n_predictors = X.shape[1]

        # Get the kernel
        kern = self.produce_kernel(n_predictors, self.ranks, self.n_out)

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
