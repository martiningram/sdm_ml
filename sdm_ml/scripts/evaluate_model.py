from sdm_ml.dataset import BBSDataset
from sdm_ml.evaluator import Evaluator
from sdm_ml.logistic_regression import LogisticRegression
from sdm_ml.single_output_gp import SingleOutputGP


dataset = BBSDataset('/Users/ingramm/Projects/uni_melb/multi_species/'
                     'bbs/dataset/csv_bird_data',
                     max_outcomes=2)

for model_name, model in [('gp', SingleOutputGP()),
                          ('log_reg', LogisticRegression())]:

    evaluator = Evaluator(dataset)
    results = evaluator.evaluate_model(model)
    results.to_csv('{}.csv'.format(model_name))
