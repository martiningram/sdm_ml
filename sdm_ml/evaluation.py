import os
from os.path import join
from functools import partial

import numpy as np
import pandas as pd
import sklearn.metrics as mt

from sdm_ml.dataset import SpeciesData
from ml_tools.evaluation import multi_class_eval
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
