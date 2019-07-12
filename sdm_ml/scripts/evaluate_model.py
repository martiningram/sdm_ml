import os
from os.path import join
import numpy as np
from tqdm import tqdm

from sdm_ml.dataset import BBSDataset
from sdm_ml.scikit_model import ScikitModel
from sdm_ml.evaluation import compute_and_save_results_for_evaluation


dataset = BBSDataset.init_using_env_variable()
training_set = dataset.training_set
test_set = dataset.test_set

experiment_name = 'new_style_run'
output_dir = join('experiments', 'new_style', experiment_name)
os.makedirs(output_dir, exist_ok=True)

models = {
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
