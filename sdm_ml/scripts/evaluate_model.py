import os
import numpy as np
from sdm_ml.dataset import BBSDataset
from sdm_ml.evaluator import Evaluator
from sdm_ml.logistic_regression import LogisticRegression
from sdm_ml.gp.single_output_gp import SingleOutputGP
from sdm_ml.gp.multi_output_gp import MultiOutputGP


dataset = BBSDataset('/Users/ingramm/Projects/uni_melb/multi_species/'
                     'bbs/dataset/csv_bird_data')

experiment_name = 'rerun_2018_9_6_rank_8_new_saving'

output_dir = os.path.join('experiments', experiment_name)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for model_name, model in [
    ('single_gp', SingleOutputGP()),
    ('multi_gp', MultiOutputGP(verbose=True, opt_steps=500, rank=8)),
    ('log_reg', LogisticRegression())
]:

    evaluator = Evaluator(dataset)
    results = evaluator.evaluate_model(model)
    results.to_csv(os.path.join(output_dir, '{}.csv'.format(model_name)))
    model.save_parameters(os.path.join(output_dir, '{}.pkl'.format(model_name)))
    names = dataset.get_training_set()['outcomes'].columns

np.save(os.path.join(output_dir, 'names'), np.array(names))
