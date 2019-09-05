import os
from os.path import join
import pandas as pd
import ml_tools.evaluation as mle
from sdm_ml.evaluation import load_all_dataset_joint_likelihoods, \
    load_all_dataset_sdm_model_results, \
    compute_joint_lik_summaries_wrt_reference


if __name__ == '__main__':

    base_dirs = [
        './experiments/experiment_summaries/to_summarise/'
    ]

    target_dir = './experiments/experiment_summaries/'
    os.makedirs(target_dir, exist_ok=True)

    reference_model = 'log_reg_cv'

    # Get all the joint likelihoods
    joint_liks = [load_all_dataset_joint_likelihoods(x) for x in base_dirs]

    # Find all the summaries
    joint_lik_summaries = pd.concat(
        [compute_joint_lik_summaries_wrt_reference(x, reference_model) for x
         in joint_liks])

    joint_lik_summaries.to_csv(join(target_dir, 'joint_lik_summaries_new.csv'))

    metrics = {'log_lik': mle.neg_log_loss_with_labels,
               'auc': mle.auc_with_nan}

    n_bootstrap = 100

    all_diffs = list()
    all_baselines = list()

    for cur_base_dir in base_dirs:

        loaded = load_all_dataset_sdm_model_results(cur_base_dir)

        for cur_dataset, cur_model_results in loaded.items():

            cur_model_results = {x: y for x, y in cur_model_results.items()
                                 if y is not None}

            # Add joint likelihood summary
            assert len(joint_liks) == 1, \
                'This code needs to be updated to handle multiple base dirs!'

            ref_joint_liks = joint_liks[0][cur_dataset][reference_model]
            mean, sd = mle.sample_mean_distribution_clt(ref_joint_liks)

            ref_y_p, ref_y_t = cur_model_results[reference_model]
            mean_ref_metrics = {
                x: mle.bootstrap_multi_class_eval_and_summarise(
                    y, ref_y_t, ref_y_p, n_bootstrap) for x, y in
                metrics.items()}

            mean_ref_metrics['joint_log_lik'] = pd.Series({
                'mean': mean,
                'sd': sd
            })

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
