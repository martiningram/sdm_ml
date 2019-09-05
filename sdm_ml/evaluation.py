import os
from os.path import join
from functools import partial

from ml_tools.paths import base_name_from_path
from glob import glob
import numpy as np
import pandas as pd
import sklearn.metrics as mt

from collections import defaultdict
from sdm_ml.dataset import SpeciesData
from ml_tools.evaluation import multi_class_eval, sample_mean_distribution_clt
from sdm_ml.presence_absence_model import PresenceAbsenceModel


def compute_and_save_results_for_evaluation(test_set: SpeciesData,
                                            model: PresenceAbsenceModel,
                                            target_dir: str) -> None:

    os.makedirs(target_dir, exist_ok=True)

    # Get log likelihood for each site
    site_lik = model.calculate_log_likelihood(
        test_set.covariates.values, test_set.outcomes.values)

    # Get marginal predictions for each site
    marg_preds = model.predict_marginal_probabilities(
        test_set.covariates.values)

    site_index = test_set.covariates.index
    species_cols = test_set.outcomes.columns

    # Save likelihoods & predictions
    pd.Series(site_lik, index=site_index).to_csv(
        join(target_dir, 'test_site_likelihoods.csv'), header=False)

    marg_preds_df = pd.DataFrame(marg_preds, index=site_index,
                                 columns=species_cols)

    marg_preds_df.to_csv(
        join(target_dir, 'marginal_species_predictions.csv'))

    # Save test outcomes for convenience
    test_set.outcomes.to_csv(join(target_dir, 'y_t.csv'))

    # Save model
    model.save_model(join(target_dir, 'trained_model'))

    metric_functions = {
        'log_loss': partial(mt.log_loss, labels=[False, True]),
        'ap_score': mt.average_precision_score,
        'auc': lambda y_t, y_p: None if len(np.unique(y_t)) == 1 else
        mt.roc_auc_score(y_t, y_p)
    }

    # Compute some summary statistics of potential interest
    marginal_metric_results = {x: multi_class_eval(
        marg_preds_df, test_set.outcomes, y, metric_name=x)
        for x, y in metric_functions.items()}

    metric_results_df = pd.DataFrame(marginal_metric_results)
    metric_results_df.to_csv(join(target_dir, 'marginal_metrics.csv'))

    # Overall summaries
    mean_metrics = metric_results_df.mean()
    mean_log_lik = np.mean(site_lik)

    mean_metrics.loc['mean_site_log_likelihood'] = mean_log_lik
    mean_metrics.to_csv(join(target_dir, 'summary_metrics.csv'))


def load_from_base_dir(base_dir, y_p_name='marginal_species_predictions',
                       y_t_name='y_t'):

    y_p_df = pd.read_csv(join(base_dir, f'{y_p_name}.csv'), index_col=0)
    y_t_df = pd.read_csv(join(base_dir, f'{y_t_name}.csv'), index_col=0)

    return y_p_df, y_t_df


def try_to_load_from_basedir(*args):

    try:
        result = load_from_base_dir(*args)
        return result
    except FileNotFoundError:
        return None


def load_sdm_ml_model_results_via_glob(base_dir):

    models = dict()

    to_use = glob(join(base_dir, '*'))

    assert all(os.path.isdir(x) for x in to_use)

    for cur_dir in to_use:

        model_name = base_name_from_path(cur_dir)
        models[model_name] = try_to_load_from_basedir(cur_dir)

    return models


def load_all_dataset_sdm_model_results(base_dir):

    subdirs = glob(join(base_dir, '*'))
    subdirs = [x for x in subdirs if os.path.isdir(x)]

    basenames = [base_name_from_path(x) for x in subdirs]

    return {x: load_sdm_ml_model_results_via_glob(y) for x, y in zip(
        basenames, subdirs)}


def load_all_dataset_joint_likelihoods(base_dir):

    # Returns all test site likelihood DataFrames in a dictionary of
    # dictionaries: dataset_name -> model_name -> df.

    subdirs = glob(join(base_dir, '*'))
    subdirs = [x for x in subdirs if os.path.isdir(x)]

    basenames = [base_name_from_path(x) for x in subdirs]

    results = defaultdict(dict)

    for cur_subdir, cur_basename in zip(subdirs, basenames):

        for cur_model_path in glob(join(cur_subdir, '*')):

            assert os.path.isdir(cur_model_path)

            cur_model_name = base_name_from_path(cur_model_path)

            cur_lik_file = join(cur_model_path, 'test_site_likelihoods.csv')

            if not os.path.isfile(cur_lik_file):
                continue

            loaded = pd.read_csv(cur_lik_file, index_col=0, header=None)[1]

            results[cur_basename][cur_model_name] = loaded

    return results


def compute_joint_lik_summaries_wrt_reference(joint_liks, reference_name):

    results = list()

    for cur_dataset, cur_model_results in joint_liks.items():

        reference_liks = cur_model_results[reference_name]

        other_models = {x: y for x, y in cur_model_results.items() if x !=
                        reference_name}

        for cur_other_model, liks in other_models.items():

            lik_diffs = liks - reference_liks

            mean, mean_sd = sample_mean_distribution_clt(lik_diffs)

            results.append({
                'dataset': cur_dataset,
                'model': cur_other_model,
                'mean_diff': mean,
                'mean_diff_sd': mean_sd
            })

    return pd.DataFrame(results)
