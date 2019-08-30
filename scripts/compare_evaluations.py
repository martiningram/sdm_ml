import os
from os.path import join
import pandas as pd
import ml_tools.evaluation as mle
from sdm_ml.evaluation import load_all_dataset_joint_likelihood_summaries
from sdm_ml.evaluation import load_all_dataset_sdm_model_results


if __name__ == '__main__':

    # TODO: Add joint log likelihood

    base_dirs = [
        './experiments/experiment_summaries/to_summarise/'
    ]

    target_dir = './experiments/experiment_summaries/'
    os.makedirs(target_dir, exist_ok=True)

    # Get the likelihoods
    results = pd.concat([load_all_dataset_joint_likelihood_summaries(x) for x
                         in base_dirs], ignore_index=True)

    results.to_csv(join(target_dir, 'joint_lik_summaries.csv'))

    metrics = {'log_lik': mle.neg_log_loss_with_labels,
               'auc': mle.auc_with_nan}

    reference_model = 'log_reg_cv'
    n_bootstrap = 1000

    all_diffs = list()
    all_baselines = list()

    for cur_base_dir in base_dirs:

        loaded = load_all_dataset_sdm_model_results(cur_base_dir)

        for cur_dataset, cur_model_results in loaded.items():

            cur_model_results = {x: y for x, y in cur_model_results.items()
                                 if y is not None}

            ref_y_p, ref_y_t = cur_model_results[reference_model]
            mean_ref_metrics = {
                x: mle.bootstrap_multi_class_eval_and_summarise(
                    y, ref_y_t, ref_y_p, n_bootstrap) for x, y in
                metrics.items()}

            mean_ref_metrics = pd.DataFrame(mean_ref_metrics)

            mean_ref_metrics['model'] = reference_model
            mean_ref_metrics['dataset'] = cur_dataset
            mean_ref_metrics['base_dir'] = cur_base_dir

            try:
                diffs = mle.compute_model_differences(
                    cur_model_results, metrics, reference_model,
                    verbose=True, n_bootstrap=n_bootstrap)
            except Exception as e:
                print(f'Unable to process {cur_dataset} because of exception:')
                print(e)
                print('Continuing.')
                continue

            diffs['dataset'] = cur_dataset
            diffs['base_dir'] = cur_base_dir

            all_diffs.append(diffs)
            all_baselines.append(mean_ref_metrics)

    all_diffs = pd.concat(all_diffs, ignore_index=True)
    all_baselines = pd.concat(all_baselines)
    all_baselines['statistic'] = all_baselines.index
    all_baselines = all_baselines.set_index('dataset', drop=False)

    all_diffs.to_csv(join(target_dir, 'bootstrap_diffs.csv'))
    all_baselines.to_csv(join(target_dir, 'reference_model_means.csv'))
