import numpy as np
import pandas as pd
import gpflow as gpf
from os.path import join
from .multi_output_gp import MultiOutputGP
from sdm_ml.presence_absence_model import PresenceAbsenceModel
from functools import partial
from distutils.dir_util import copy_tree


def select_using_standard_error_rule(means: np.ndarray,
                                     stderrs: np.ndarray) -> int:
    """Selects result according to one standard error rule (see ESL p.244).

    Args:
        means: Mean errors obtained by cross validation [lower is better].
        stderrs: Standard errors obtained from cross validation.

    Note:
        The arrays are assumed to be sorted by ascending complexity of the
        model used to fit them.

    Returns:
        The index of the element within a standard deviation of the best
        mean error.
    """

    indices = np.arange(len(means))

    best_mean_idx = np.argmin(means)
    best_mean, best_stderr = means[best_mean_idx], stderrs[best_mean_idx]

    max_acceptable = best_mean + best_stderr
    less_complex = means[:best_mean_idx]
    remaining_indices = indices[:best_mean_idx]

    # If the best mean is the smallest, nothing to do
    if less_complex.shape[0] == 0:
        return best_mean_idx

    # Otherwise, assess the less complex results
    relevant_indices = remaining_indices[less_complex < max_acceptable]

    if relevant_indices.shape[0] == 0:
        return best_mean_idx
    else:
        return relevant_indices.min()


class CrossValidatedMultiOutputGP(PresenceAbsenceModel):

    def __init__(self, variances_to_try, cv_save_dir, n_folds=4, n_kernels=10,
                 add_bias=True, rbf_var=0.1, bias_var=0.1,
                 kern_var_trainable=False, n_inducing=100, maxiter=int(1E6)):

        self.model = None
        self.is_fit = False
        self.variances_to_try = variances_to_try
        self.cv_save_dir = cv_save_dir
        self.n_folds = n_folds

        self.kernel_fun = partial(
            MultiOutputGP.build_default_kernel, n_kernels=n_kernels,
            add_bias=add_bias, kern_var_trainable=kern_var_trainable,
            rbf_var=rbf_var)

        self.model_fun = partial(MultiOutputGP, n_inducing=n_inducing,
                                 n_latent=n_kernels, maxiter=maxiter)

    def fit(self, X, y):

        n_dims = X.shape[1]
        n_outputs = y.shape[1]

        kern_fun = partial(self.kernel_fun, n_dims=n_dims, n_outputs=n_outputs)

        def get_model(w_prior, bias_var):

            # We need to make a model creation function.
            cur_kernel = kern_fun(w_prior=w_prior, bias_var=bias_var)
            model_fun = partial(self.model_fun, kernel=cur_kernel)
            return model_fun()

        scores = list()

        for cur_variance in self.variances_to_try:

            # Compute the bias variance so that we have a variance of 0.4
            # for that overall
            bias_var = 0.4 / cur_variance

            print(f'Fitting {cur_variance:.2f} with bias var {bias_var:.2f}')

            model_fun = lambda: get_model(cur_variance, bias_var) # NOQA

            cur_mean_score, cur_stderr = MultiOutputGP.cross_val_score(
                X, y, model_fun, save_dir=join(
                    self.cv_save_dir, f'{cur_variance:.2f}'),
                n_folds=self.n_folds)

            gpf.reset_default_graph_and_session()

            print(f'Mean likelihood is {cur_mean_score}')

            scores.append({'mean': cur_mean_score, 'stderr': cur_stderr,
                           'variance': cur_variance})

        scores = pd.DataFrame(scores)

        # Sort by ascending complexity
        scores.sort_values('variance')

        # Find best index; invert mean since error rule expects errors,
        # where smaller is better, rather than likelihoods where higher is
        # better.
        best_idx = select_using_standard_error_rule(
            -scores['mean'].values, scores['stderr'].values)

        best_variance = scores.iloc[best_idx]['variance']

        print(f'Selected model using one standard error rule has variance '
              f' {best_variance:.2f}')

        bias_var = 0.4 / best_variance

        best_model = get_model(best_variance, bias_var)

        best_model.fit(X, y)

        self.is_fit = True
        self.model = best_model

    def predict_log_marginal_probabilities(self, X):

        return self.model.predict_log_marginal_probabilities(X)

    def calculate_log_likelihood(self, X, y):

        return self.model.calculate_log_likelihood(X, y)

    def save_model(self, target_folder):

        self.model.save_model(target_folder)

        # Copy over the cv results
        copy_tree(self.cv_save_dir, join(target_folder, 'cv_results'))
