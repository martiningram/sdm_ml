import os
import numpy as np
from sdm_ml.dataset import BBSDataset
from sdm_ml.evaluator import Evaluator
from sdm_ml.logistic_regression import LogisticRegression
from sdm_ml.gp.single_output_gp import SingleOutputGP
from sdm_ml.gp.multi_output_gp import MultiOutputGP
from sdm_ml.warton_model import WartonModel
from sdm_ml.maps import produce_maps


dataset = BBSDataset('/Users/ingramm/Projects/uni_melb/multi_species/'
                     'bbs/dataset/csv_bird_data',
                     max_outcomes=2)

experiment_name = 'warton'

output_dir = os.path.join('experiments', experiment_name)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for model_name, model in [
#     ('multi_gp', MultiOutputGP(verbose=True, opt_steps=500, rank=8,
#                                fixed_lengthscales=np.array(
#                                    [8.61,  5.29,  2.72, 13.08, 13.6 ,  7.77,
#                                     14.  ,  2.93]))),
    ('log_reg', LogisticRegression()),
    ('warton', WartonModel(n_latents=2))
]:

    evaluator = Evaluator(dataset)
    results = evaluator.evaluate_model(model)
    results.to_csv(os.path.join(output_dir, '{}.csv'.format(model_name)))
    model.save_parameters(os.path.join(output_dir, model_name))
    names = dataset.get_training_set()['outcomes'].columns

    # Make maps
    produce_maps(dataset, model, os.path.join(
        output_dir, '{}_maps'.format(model_name)))

np.save(os.path.join(output_dir, 'names'), np.array(names))
