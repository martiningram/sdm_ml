import os
from sdm_ml.dataset import BBSDataset
from sdm_ml.evaluator import Evaluator
from sdm_ml.logistic_regression import LogisticRegression
from sdm_ml.gp.single_output_gp import SingleOutputGP
from sdm_ml.gp.multi_output_gp import MultiOutputGP


dataset = BBSDataset('/Users/ingramm/Projects/uni_melb/multi_species/'
                     'bbs/dataset/csv_bird_data', max_outcomes=3)

experiment_name = 'multi_gp'

output_dir = os.path.join('experiments', experiment_name)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for model_name, model in [('multi_gp', MultiOutputGP(verbose=True)),
                          ('gp', SingleOutputGP()),
                          ('log_reg', LogisticRegression())]:

    evaluator = Evaluator(dataset)
    results = evaluator.evaluate_model(model)
    results.to_csv(os.path.join(output_dir, '{}.csv'.format(model_name)))
