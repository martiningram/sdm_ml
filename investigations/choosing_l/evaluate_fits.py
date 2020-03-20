import numpy as np
import pandas as pd
from svgp.tf.models.mogp_classifier import (
    predict_probs, restore_results, predict_f_samples)
from ml_tools.utils import load_pickle_safely
from os.path import join
import sys
from ml_tools.evaluation import multi_class_eval
from ml_tools.normals import calculate_log_joint_bernoulli_likelihood
from sklearn.metrics import roc_auc_score, log_loss
from os import makedirs
import json
from tqdm import tqdm


base_dir = sys.argv[1]
test_cov_file = sys.argv[2]
test_outcomes_file = sys.argv[3]
target_dir = sys.argv[4]
n_draws_predict = 1000

makedirs(target_dir, exist_ok=True)

sample_results = np.load(join(base_dir, 'results.npz'))
scaler = load_pickle_safely(join(base_dir, 'scaler.pkl'))
settings = json.load(open(join(base_dir, 'settings.json')))

mogp_results = restore_results(sample_results)

test_covs = pd.read_csv(test_cov_file, index_col=0)
test_y = pd.read_csv(test_outcomes_file, index_col=0)

X_test = scaler.transform(test_covs.values)
prob_pred = predict_probs(mogp_results, X_test)
prob_pred = pd.DataFrame(prob_pred, columns=test_y.columns)

prob_pred.to_csv(join(target_dir, 'probs_pred.csv'))

metric_dict = {
    'auc': roc_auc_score,
    'log_loss': log_loss
}

per_species_results = dict()
mean_results = dict()

for cur_name, cur_metric_fn in metric_dict.items():

    cur_eval_results = multi_class_eval(prob_pred, test_y, cur_metric_fn)
    cur_mean_results = cur_eval_results.mean()

    per_species_results[cur_name] = cur_eval_results
    mean_results[cur_name] = cur_mean_results

pd.DataFrame(per_species_results).to_csv(
    join(target_dir, 'per_species_metrics.csv'))

mean_results.update(settings)

# Also predict samples for the joint likelihood
site_iterator = predict_f_samples(mogp_results, X_test, n_draws_predict)

joint_log_liks = np.zeros(X_test.shape[0])

for i, cur_site_draws in enumerate(tqdm(site_iterator)):

    joint_log_liks[i] = calculate_log_joint_bernoulli_likelihood(
        cur_site_draws, test_y.values[i])

pd.Series(joint_log_liks).to_csv(join(target_dir, 'site_joint_liks.csv'))

mean_results['joint_log_lik'] = joint_log_liks.mean()

pd.Series(mean_results).to_csv(join(target_dir, 'summary_metrics.csv'))
