import os
import time
import gpflow
import numpy as np
import pandas as pd
from ml_tools.utils import save_pickle_safely
from os.path import join
from functools import partial
from sdm_ml.dataset import BBSDataset
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from ml_tools.utils import create_path_with_variables
from GPyOpt.methods import BayesianOptimization


base_save_dir = './experiments/cv_gpflow/'
test_run = False
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
    bopt_maxiter = 2
else:
    n_folds = 4
    maxiter = int(1E6)
    bopt_maxiter = 50


def create_model(n_kernels, n_inducing, add_bias, mean_function):

    kern = MultiOutputGP.build_default_kernel(
        n_dims=X.shape[1], n_kernels=n_kernels, n_outputs=y.shape[1],
        add_bias=add_bias)

    return MultiOutputGP(n_inducing=n_inducing, n_latent=n_kernels,
                         kernel=kern, maxiter=maxiter,
                         n_draws_predict=int(1E3),
                         mean_function=mean_function)


domain = [
    {'name': 'n_kernels', 'type': 'discrete', 'domain': (2, 3, 4, 5, 6, 7, 8)},
    {'name': 'n_inducing', 'type': 'discrete',
     'domain': [5, 10, 20, 30, 50, 100]},
    {'name': 'add_bias', 'type': 'discrete', 'domain': [0, 1]},
    {'name': 'mean_function', 'type': 'discrete', 'domain':  [0, 1]}]

STEP = 0

# Add on the date
base_save_dir = join(base_save_dir,
                     create_path_with_variables(test_run=test_run,
                                                method='bayesopt',
                                                shuffle=shuffle))

os.makedirs(base_save_dir, exist_ok=True)

live_file = open(join(base_save_dir, 'cv_scores_live.csv'), 'a', buffering=1)


def to_optimise(x):

    global STEP

    # Get model
    x = x[0]

    gpflow.reset_default_graph_and_session()

    n_kernels, n_inducing, add_bias, use_mean_function = map(int, x)

    add_bias = add_bias == 1
    use_mean_function = use_mean_function == 1

    cur_save_dir = join(base_save_dir, create_path_with_variables(
        n_kernels=n_kernels, n_inducing=n_inducing, add_bias=add_bias,
        use_mean_function=use_mean_function))

    if use_mean_function:
        mean_function = partial(MultiOutputGP.build_default_mean_function,
                                n_outputs=y.shape[1])
    else:
        mean_function = lambda: None # NOQA

    model_fn = partial(create_model, n_kernels=n_kernels,
                       n_inducing=n_inducing, add_bias=add_bias,
                       mean_function=mean_function)

    start_time = time.time()

    lik_mean = MultiOutputGP.cross_val_score(X, y, model_fn, cur_save_dir,
                                             n_folds=n_folds)

    # Store
    cur_result = {'n_kernels': n_kernels, 'n_inducing': n_inducing,
                  'log_lik': lik_mean, 'runtime': time.time() - start_time,
                  'n_folds': n_folds, 'add_bias': add_bias,
                  'mean_function': use_mean_function, 'step': STEP}

    cur_result = pd.DataFrame([cur_result])
    cur_result.to_csv(join(cur_save_dir, 'results.csv'))

    write_header = STEP == 0

    cur_result.to_csv(live_file, header=write_header, index=False)

    STEP += 1

    # GPyOpt likes to minimise
    return -lik_mean


myBopt = BayesianOptimization(f=to_optimise, domain=domain)
myBopt.run_optimization(max_iter=bopt_maxiter,
                        report_file=join(base_save_dir, 'report.txt'),
                        evaluations_file=join(base_save_dir, 'evals.txt'),
                        models_file=join(base_save_dir, 'models.txt'))

# Store bayesian optimisation
live_file.close()
save_pickle_safely(myBopt, join(base_save_dir, 'bayes_opt_object.pkl'))
