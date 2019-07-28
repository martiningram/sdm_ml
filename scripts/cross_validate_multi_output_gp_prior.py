import os
import time
import gpflow
import pandas as pd
import numpy as np
from os.path import join
from functools import partial
from sdm_ml.dataset import BBSDataset
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from ml_tools.utils import create_path_with_variables


base_save_dir = './experiments/cv_gpflow/'
test_run = True
shuffle = True
seed = 1

np.random.seed(seed)

dataset = BBSDataset(os.environ['BBS_PATH'])

X = dataset.training_set.covariates.values
y = dataset.training_set.outcomes.values.astype(int)

if shuffle:

    order = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
    X = X[order]
    y = y[order]

if test_run:
    X = X[:200]
    y = y[:200, :10]
    n_folds = 2
    maxiter = 100
else:
    n_folds = 4
    maxiter = int(1E6)


def create_model(w_prior):

    kern = MultiOutputGP.build_default_kernel(
        n_dims=X.shape[1], n_kernels=10, n_outputs=y.shape[1],
        add_bias=True, w_prior=w_prior)

    return MultiOutputGP(n_inducing=100, n_latent=10, kernel=kern,
                         maxiter=maxiter, n_draws_predict=int(1E4))


grid = {'w_prior': np.linspace(0.1, 1., 5)**2}

# Add on the date
base_save_dir = join(base_save_dir,
                     create_path_with_variables(test_run=test_run,
                                                shuffle=shuffle))

i = 0

scores = list()

for cur_w_prior in grid['w_prior']:

    gpflow.reset_default_graph_and_session()

    start_time = time.time()

    print(f'On iteration {i}.')

    print(f'Running with w_prior={cur_w_prior}.')

    model_fn = partial(create_model, w_prior=cur_w_prior)

    save_dir = create_path_with_variables(w_prior=cur_w_prior,
                                          test_run=test_run)

    cur_score = MultiOutputGP.cross_val_score(X, y, model_fn, join(
        base_save_dir, save_dir), n_folds=n_folds)

    scores.append({
        'log_lik': cur_score,
        'runtime': time.time() - start_time,
        'n_folds': n_folds,
        'w_prior': cur_w_prior
    })

    pd.DataFrame(scores).to_csv(join(base_save_dir,
                                        'cv_results_live.csv'))

    pd.Series(scores[-1]).to_csv(join(base_save_dir, save_dir,
                                        'scores.csv'))

    print(f'Iteration {i} result: {scores[-1]}')

    i += 1

scores = pd.DataFrame(scores)
scores.to_csv(join(base_save_dir, 'cv_results.csv'))
