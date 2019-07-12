import os
import time
import pandas as pd
from os.path import join
from functools import partial
from sdm_ml.dataset import BBSDataset
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from ml_tools.utils import create_path_with_variables


base_save_dir = './experiments/cv_gpflow/'
test_run = True

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


def create_model(n_kernels, n_inducing):

    kern = MultiOutputGP.build_default_kernel(
        n_dims=X.shape[1], n_kernels=n_kernels, n_outputs=y.shape[1])

    return MultiOutputGP(n_inducing=n_inducing, n_latent=n_kernels,
                         kernel=kern, maxiter=maxiter)


grid = {'n_kernels': [2, 4, 6, 8], 'n_inducing': [20, 100]}

# Add on the date
base_save_dir = join(base_save_dir,
                     create_path_with_variables(test_run=test_run))

i = 0

scores = list()

for cur_n_kernels in grid['n_kernels']:

    for cur_n_inducing in grid['n_inducing']:

        start_time = time.time()

        print(f'On iteration {i}.')

        print(f'Running with {cur_n_kernels} kernels and {cur_n_inducing}'
              f' inducing.')

        model_fn = partial(create_model, n_kernels=cur_n_kernels,
                           n_inducing=cur_n_inducing)

        save_dir = create_path_with_variables(L=cur_n_kernels,
                                              M=cur_n_inducing)

        cur_score = MultiOutputGP.cross_val_score(X, y, model_fn, join(
            base_save_dir, save_dir), n_folds=n_folds)

        scores.append({
            'n_kernels': cur_n_kernels,
            'n_inducing': cur_n_inducing,
            'log_lik': cur_score,
            'runtime': time.time() - start_time,
            'n_folds': n_folds
        })

        pd.DataFrame(scores).to_csv(join(base_save_dir, 'cv_results_live.csv'))

        pd.Series(scores[-1]).to_csv(join(base_save_dir, save_dir,
                                          'scores.csv'))

        print(f'Iteration {i} result: {scores[-1]}')

        i += 1

scores = pd.DataFrame(scores)
scores.to_csv(join(base_save_dir, 'cv_results.csv'))