import os
import numpy as np
from os.path import join
from tqdm import tqdm
from functools import partial

from sdm_ml.dataset import BBSDataset
from sdm_ml.scikit_model import ScikitModel
from sdm_ml.evaluation import compute_and_save_results_for_evaluation
from sdm_ml.gp.single_output_gp import SingleOutputGP


test_run = True
max_outcomes = 3 if test_run else None

dataset = BBSDataset.init_using_env_variable(max_outcomes=max_outcomes)
training_set = dataset.training_set
test_set = dataset.test_set

experiment_name = 'new_style_run_test'
output_dir = join('experiments', 'new_style', experiment_name)
os.makedirs(output_dir, exist_ok=True)

# Fetch single output GP
default_kernel_fun = partial(SingleOutputGP.build_default_kernel,
                             n_dims=training_set.covariates.shape[1],
                             add_bias=True)

so_gp = SingleOutputGP(n_inducing=100, kernel_function=default_kernel_fun,
                       verbose_fit=False,
                       maxiter=10 if test_run else int(1E6))

models = {
    'Single Output GP': so_gp,
    'RandomForestCV': ScikitModel(ScikitModel.create_cross_validated_forest),
    'LogRegCV': ScikitModel(),
}

for cur_name, cur_model in tqdm(models.items()):

    cur_output_dir = join(output_dir, cur_name)
    cur_model.fit(training_set.covariates.values,
                  training_set.outcomes.values.astype(int))
    compute_and_save_results_for_evaluation(test_set, cur_model,
                                            cur_output_dir)

np.save(join(output_dir, 'names'), test_set.outcomes.columns.values)
