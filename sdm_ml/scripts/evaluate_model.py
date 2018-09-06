import os
import numpy as np
from sdm_ml.dataset import BBSDataset
from sdm_ml.evaluator import Evaluator
from sdm_ml.logistic_regression import LogisticRegression
from sdm_ml.gp.single_output_gp import SingleOutputGP
from sdm_ml.gp.multi_output_gp import MultiOutputGP


dataset = BBSDataset('/Users/ingramm/Projects/uni_melb/multi_species/'
                     'bbs/dataset/csv_bird_data')

experiment_name = 'multi_gp_rank_4_rerun_2'

output_dir = os.path.join('experiments', experiment_name)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

for model_name, model in [
    ('multi_gp', MultiOutputGP(verbose=True, opt_steps=500, rank=4))]:

    evaluator = Evaluator(dataset)
    results = evaluator.evaluate_model(model)
    results.to_csv(os.path.join(output_dir, '{}.csv'.format(model_name)))

    # Save the model, too
    main_kernel = model.model.kern.children['kernels'][0]
    coreg_kernel = model.model.kern.children['kernels'][1]

    main_lengthscales = main_kernel.lengthscales.value
    main_variance = main_kernel.variance.value
    coreg_w = coreg_kernel.W.value
    coreg_kappa = coreg_kernel.kappa.value
    inducing = model.model.feature.Z.value

    np.savez(os.path.join(output_dir, 'kernel_params'),
             coreg_w=coreg_w, coreg_kappa=coreg_kappa,
             lengthscales=main_lengthscales,
             main_variance=main_variance,
             inducing=inducing)
