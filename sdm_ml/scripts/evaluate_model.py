import os
import numpy as np
from sdm_ml.dataset import BBSDataset
from sdm_ml.evaluator import Evaluator
from sdm_ml.logistic_regression import LogisticRegression
from sdm_ml.gp.single_output_gp import SingleOutputGP
from sdm_ml.gp.icm_multi_gp import ICMMultiGP
from sdm_ml.gp.lmc_multi_gp import LMCMultiGP
from sdm_ml.gp.new_style_multi_gp import NewStyleMultiGP
from sdm_ml.warton_model import WartonModel
from sdm_ml.gp.warton_gp import WartonGP
from sdm_ml.maps import produce_maps


dataset = BBSDataset('/Users/ingramm/Projects/uni_melb/multi_species/'
                     'bbs/dataset/csv_bird_data',
                     max_outcomes=32)

experiment_name = 'warton_gp_test'

output_dir = os.path.join('experiments', experiment_name)

make_maps = False

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for model_name, model in [
    ('warton_gp', WartonGP(verbose=True, opt_steps=500, rank=4))
]:

    evaluator = Evaluator(dataset)
    results = evaluator.evaluate_model(model)
    results.to_csv(os.path.join(output_dir, '{}.csv'.format(model_name)))
    model.save_parameters(os.path.join(output_dir, model_name))
    names = dataset.get_training_set()['outcomes'].columns

    # Make maps
    if make_maps:
        produce_maps(dataset, model, os.path.join(
            output_dir, '{}_maps'.format(model_name)))

np.save(os.path.join(output_dir, 'names'), np.array(names))
