import os
import time
import gpflow
import pandas as pd
from os.path import join
from functools import partial
from sdm_ml.dataset import BBSDataset
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from ml_tools.utils import create_path_with_variables


base_save_dir = './experiments/cv_gpflow/'
test_run = False

dataset = BBSDataset(os.environ['BBS_PATH'])

X = dataset.training_set.covariates.values
y = dataset.training_set.outcomes.values.astype(int)

if test_run:
    X = X[:200]
    y = y[:200, :10]
    n_folds = 2
    maxiter = 100
else:
    n_folds = 4
    maxiter = int(1E6)


def create_model(n_kernels, n_inducing, add_bias):

    kern = MultiOutputGP.build_default_kernel(
        n_dims=X.shape[1], n_kernels=n_kernels, n_outputs=y.shape[1],
        add_bias=add_bias)

    return MultiOutputGP(n_inducing=n_inducing, n_latent=n_kernels,
                         kernel=kern, maxiter=maxiter,
                         n_draws_predict=int(1E3))


grid = {'n_kernels': [2, 4, 6, 8], 'n_inducing': [10, 20, 50],
        'add_bias': [False, True]}

# Add on the date
base_save_dir = join(base_save_dir,
                     create_path_with_variables(test_run=test_run))

i = 0

scores = list()

for cur_add_bias in grid['add_bias']:

    for cur_n_kernels in grid['n_kernels']:

        for cur_n_inducing in grid['n_inducing']:

            gpflow.reset_default_graph_and_session()

            start_time = time.time()

            print(f'On iteration {i}.')

            print(f'Running with {cur_n_kernels} kernels and {cur_n_inducing}'
                  f' inducing and bias={cur_add_bias}.')

            model_fn = partial(create_model, n_kernels=cur_n_kernels,
                               n_inducing=cur_n_inducing,
                               add_bias=cur_add_bias)

            save_dir = create_path_with_variables(
                L=cur_n_kernels, M=cur_n_inducing, add_bias=cur_add_bias,
                comment='new_priors_17-07-19')

            cur_score = MultiOutputGP.cross_val_score(X, y, model_fn, join(
                base_save_dir, save_dir), n_folds=n_folds)

            scores.append({
                'n_kernels': cur_n_kernels,
                'n_inducing': cur_n_inducing,
                'log_lik': cur_score,
                'runtime': time.time() - start_time,
                'n_folds': n_folds,
                'add_bias': cur_add_bias
            })

            pd.DataFrame(scores).to_csv(join(base_save_dir,
                                             'cv_results_live.csv'))

            pd.Series(scores[-1]).to_csv(join(base_save_dir, save_dir,
                                            'scores.csv'))

            print(f'Iteration {i} result: {scores[-1]}')

            i += 1

scores = pd.DataFrame(scores)
scores.to_csv(join(base_save_dir, 'cv_results.csv'))
