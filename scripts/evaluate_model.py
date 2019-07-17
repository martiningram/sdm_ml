import os
import numpy as np
from os.path import join
from tqdm import tqdm
from functools import partial

from sdm_ml.dataset import BBSDataset
from sdm_ml.scikit_model import ScikitModel
from sdm_ml.evaluation import compute_and_save_results_for_evaluation
from sdm_ml.gp.single_output_gp import SingleOutputGP
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from ml_tools.utils import create_path_with_variables


test_run = False
max_outcomes = 5 if test_run else None

dataset = BBSDataset.init_using_env_variable(max_outcomes=max_outcomes)
training_set = dataset.training_set
test_set = dataset.test_set

experiment_name = create_path_with_variables(test_run=test_run)
output_dir = join('experiments', 'new_style', experiment_name)
os.makedirs(output_dir, exist_ok=True)

# Fetch single output GP
default_kernel_fun = partial(SingleOutputGP.build_default_kernel,
                             n_dims=training_set.covariates.shape[1],
                             add_bias=True)

so_gp = SingleOutputGP(n_inducing=100, kernel_function=default_kernel_fun,
                       verbose_fit=False,
                       maxiter=10 if test_run else int(1E6))

# Fetch multi output GP
n_kernels = 8
mogp_kernel = MultiOutputGP.build_default_kernel(
    n_dims=training_set.covariates.shape[1],
    n_kernels=n_kernels, n_outputs=training_set.outcomes.shape[1],
    add_bias=True, priors=True)

mogp = MultiOutputGP(n_inducing=100, n_latent=n_kernels, kernel=mogp_kernel,
                     maxiter=10 if test_run else int(1E6))


models = {
    'Multi Output GP': mogp,
    'RandomForestCV': ScikitModel(ScikitModel.create_cross_validated_forest),
    'LogRegCV': ScikitModel(),
    'Single Output GP': so_gp,
}

np.save(join(output_dir, 'names'), test_set.outcomes.columns.values)

for cur_name, cur_model in tqdm(models.items()):

    cur_output_dir = join(output_dir, cur_name)
    cur_model.fit(training_set.covariates.values,
                  training_set.outcomes.values.astype(int))
    compute_and_save_results_for_evaluation(test_set, cur_model,
                                            cur_output_dir)
