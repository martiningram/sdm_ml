import os
import pandas as pd
from glob import glob
from python_tools.paths import base_name_from_path
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from scipy.special import logit
from os.path import join
from ml_tools.stan import load_stan_model_cached
from sklearn.preprocessing import LabelEncoder


def bootstrap_metric(metric, y_t, y_p, n_samples=1000):

    boot_metrics = np.empty(n_samples)

    for i in range(n_samples):

        # Get a bootstrap sample
        sample = np.random.choice(y_t.shape[0], size=y_t.shape[0],
                                  replace=True)

        boot_y_t = y_t[sample]
        boot_y_p = y_p[sample]

        cur_metric = metric(boot_y_t, boot_y_p)

        boot_metrics[i] = cur_metric

    return boot_metrics


def bootstrap_many(metric, y_t_df, y_p_df, n_samples=1000):

    results = dict()

    for cur_outcome in y_t_df.columns:

        cur_y_t = y_t_df[cur_outcome].values
        cur_y_p = y_p_df[cur_outcome].values

        cur_metrics = bootstrap_metric(metric, cur_y_t, cur_y_p,
                                       n_samples=n_samples)

        results[cur_outcome] = cur_metrics

    return pd.DataFrame(results)


def load_from_base_dir(base_dir, y_p_name='marginal_species_predictions',
                       y_t_name='y_t'):

    y_p_df = pd.read_csv(join(base_dir, f'{y_p_name}.csv'), index_col=0)
    y_t_df = pd.read_csv(join(base_dir, f'{y_t_name}.csv'), index_col=0)

    return y_p_df, y_t_df


def get_logit_metric_summaries(metric_fn, y_t_df, y_p_df):

    logit_metric = lambda y_t, y_p: logit(metric_fn(y_t, y_p)) # NOQA
    full_results = bootstrap_many(logit_metric, y_t_df, y_p_df)

    return full_results, full_results.mean(), full_results.std()


def get_raw_metric_summaries(metric_fn, y_t_df, y_p_df):

    full_results = bootstrap_many(metric_fn, y_t_df, y_p_df)

    return full_results, full_results.mean(), full_results.std()


def load_sdm_ml_model_results_via_glob(base_dir):

    models = dict()

    to_use = glob(join(base_dir, '*'))

    assert all(os.path.isdir(x) for x in to_use)

    for cur_dir in to_use:

        model_name = base_name_from_path(cur_dir)

        # Default args should do here
        models[model_name] = load_from_base_dir(cur_dir)

    return models


def calculate_model_bootstrap_results(
        models, metric_fn=average_precision_score):

    print('Calculating bootstrap results...')

    bootstrap_results = list()

    for cur_name, (cur_y_p, cur_y_t) in tqdm(models.items()):

        print(f'Bootstrapping {cur_name}...')

        if test_run:
            cur_y_p = cur_y_p.iloc[:, :2]
            cur_y_t = cur_y_t.iloc[:, :2]

        summary_full, means, sds = get_logit_metric_summaries(
            metric_fn, cur_y_t, cur_y_p)

        combined = pd.concat([means, sds], axis=1, keys=['mean', 'sd'])
        combined['model'] = cur_name
        combined['species'] = cur_y_t.columns

        bootstrap_results.append(combined)

    bootstrap_results = pd.concat(bootstrap_results, ignore_index=True)

    return bootstrap_results


def prepare_stan_data(bootstrap_df):

    model_encoder = LabelEncoder()
    species_encoder = LabelEncoder()

    model_id = model_encoder.fit_transform(bootstrap_df['model']) + 1
    species_id = species_encoder.fit_transform(bootstrap_df['species']) + 1

    model_data = {
        'N': bootstrap_df.shape[0],
        'n_species': len(species_encoder.classes_),
        'n_models': len(model_encoder.classes_),
        'model_id': model_id,
        'species_id': species_id,
        'metric_mean': bootstrap_df['mean'].values,
        'metric_sd': bootstrap_df['sd'].values
    }

    return model_data, model_encoder, species_encoder


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', required=True, type=str)
    parser.add_argument('--metric', required=True, type=str)
    parser.add_argument('--test-run', required=False, action='store_true')
    parser.add_argument('--target-base-dir', required=False, type=str,
                        default='./bootstrap_mixed_result')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    test_run = args.test_run
    metric = args.metric
    target_dir = join(args.target_base_dir, f'/{dataset_name}/{metric}/')

    os.makedirs(target_dir, exist_ok=True)

    metric_lookup = {
        'ap': average_precision_score,
        'auc': roc_auc_score,
    }

    metric_fn = metric_lookup[metric]

    models = {
        'mixed': load_from_base_dir(f'../../R/datasets/{dataset_name}',
                                    'mixed_predictions', 'y_test'),
        'brt': load_from_base_dir(f'../../R/datasets/{dataset_name}',
                                  'brt_predictions', 'y_test')
    }

    other_models = load_sdm_ml_model_results_via_glob(
        f'./experiment_results/test_run=False, 2019-07-30 10:44:49.011662/'
        f'{dataset_name}')

    models.update(other_models)

    cv_models = load_sdm_ml_model_results_via_glob(
        f'./experiment_results/test_run=False, 2019-08-01 16:16:25.346028/'
        f'{dataset_name}')

    models.update(cv_models)

    # TODO: Maybe add multiple metrics here
    bootstrap_results = calculate_model_bootstrap_results(
        models, metric_fn=metric_fn)

    bootstrap_results.to_csv(
        join(target_dir, f'bootstrap_summaries.csv'))

    # Filter for NaNs
    to_drop = bootstrap_results.groupby('model').apply(
        lambda df: df[df['sd'].isnull()].species).values
    to_drop = set(to_drop)
    filtered = bootstrap_results[~bootstrap_results['species'].isin(to_drop)]
    filtered.groupby('model')['species'].count()

    filtered.to_csv(
        join(target_dir, f'bootstrap_summaries_filtered.csv'))

    stan_model = load_stan_model_cached('./metric_model_normal.stan')
    model_data, model_encoder, species_encoder = prepare_stan_data(filtered)
    fit = stan_model.sampling(data=model_data)

    with open(join(target_dir, f'stan_fit.txt'), 'w') as f:

        print(fit, file=f)

    np.savez(join(target_dir, f'stan_arrays'), **fit.extract())

    model_effect_draws = pd.DataFrame(
        fit['model_effect'], columns=model_encoder.classes_)

    stacked_other_draws = pd.DataFrame(
        {'intercept': fit['intercept'],
         'species_effect_sd': fit['species_effect_sd']})

    model_draws = pd.concat([model_effect_draws, stacked_other_draws], axis=1)

    model_draws.to_csv(join(target_dir, f'draws.csv'))
