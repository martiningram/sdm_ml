import numpy as np
from os import makedirs
from os.path import join
from sdm_ml.dataset import BBSDataset
from sdm_ml.scikit_model import ScikitModel
from svgp.tf.mogp_classifier import (
    restore_results, predict_probs as predict_probs_mogp)
from svgp.tf.sogp_classifier import predict_probs, load_results
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from ml_tools.plotting import conditional_plot_2d


dataset = BBSDataset.init_using_env_variable()
train_covariates = dataset.training_set.covariates.values
train_outcomes = dataset.training_set.outcomes
scaler = StandardScaler()

scaled_covs = scaler.fit_transform(train_covariates)

target_dir = './experiments/conditional_plots/'
makedirs(target_dir, exist_ok=True)

mogp_file = ('./experiments/experiment_summaries/to_summarise/bbs/'
             'hierarchical_mogp_10_gamma32/trained_model/results_file.npz')

sogp_dir = ('./experiments/experiment_summaries/to_summarise/bbs/'
            'sogp_new/trained_model/')

species_1 = 'Cerulean Warbler'
species_2 = "Scarlet Tanager"

cov_1 = 0
cov_2 = 4

cov_1_name = dataset.training_set.covariates.columns[cov_1]
cov_2_name = dataset.training_set.covariates.columns[cov_2]
n_covs = dataset.training_set.covariates.shape[1]

pred_funs_1 = dict()
pred_funs_2 = dict()

y1 = train_outcomes[species_1].values
y2 = train_outcomes[species_2].values

n_obs_1 = y1.sum()
n_obs_2 = y2.sum()

loaded_mogp = np.load(mogp_file)
mogp_results = restore_results(loaded_mogp)

all_species = dataset.training_set.outcomes.columns

# Make sure these exist
assert species_1 in all_species and species_2 in all_species

# Find their ids
species_1_num = np.argmax(all_species == species_1)
species_2_num = np.argmax(all_species == species_2)

# Load the models
# We'll just refit the RF and the logistic regression since they're quick
# for two species:
rf_model_1 = ScikitModel.create_cross_validated_forest(
    n_covariates=train_covariates.shape[1])

rf_model_2 = ScikitModel.create_cross_validated_forest(
    n_covariates=train_covariates.shape[1])

print('Fitting RF...')
rf_model_1.fit(scaled_covs, y1)
rf_model_2.fit(scaled_covs, y2)
print('Done.')

pred_funs_1['rf'] = lambda x: rf_model_1.predict_log_proba(x)[:, 1]
pred_funs_2['rf'] = lambda x: rf_model_2.predict_log_proba(x)[:, 1]

# Load and fit the logistic regression too
log_reg_model_1 = LogisticRegression(penalty='none', solver='newton-cg')
log_reg_model_2 = LogisticRegression(penalty='none', solver='newton-cg')

log_reg_model_1.fit(scaled_covs, y1)
log_reg_model_2.fit(scaled_covs, y2)

pred_funs_1['log_reg'] = lambda x: log_reg_model_1.predict_log_proba(x)[:, 1]
pred_funs_2['log_reg'] = lambda x: log_reg_model_2.predict_log_proba(x)[:, 1]

# Load the SOGP too
sogp_model_1 = load_results(
    join(sogp_dir, f'results_file_{species_1_num}.npz'))

sogp_model_2 = load_results(
    join(sogp_dir, f'results_file_{species_2_num}.npz'))

pred_funs_1['sogp'] = lambda x: predict_probs(sogp_model_1, x, log=True)
pred_funs_2['sogp'] = lambda x: predict_probs(sogp_model_2, x, log=True)

# And predict using the MOGP
pred_funs_1['mogp'] = lambda x: predict_probs_mogp(mogp_results, x, log=True)[
    :, species_1_num]

pred_funs_2['mogp'] = lambda x: predict_probs_mogp(mogp_results, x, log=True)[
    :, species_2_num]

# Now that we have all the prediction functions in place, make the partial
# plots
# TODO: Predicting the probs for the MOGP is slow. Maybe reduce the number of
# draws or points in the contour plot?
for cur_model_name, cur_pred_fun in pred_funs_1.items():

    print(f'Predicting species 1 for {cur_model_name}')

    cur_x, cur_y, cur_z = conditional_plot_2d(
        cur_pred_fun, [cov_1, cov_2], n_covs, n_points=100)

    np.savez(join(target_dir, f'{cur_model_name}_1'),
             x=cur_x, y=cur_y, z=cur_z, model_name=cur_model_name,
             cov_1_name=cov_1_name, cov_2_name=cov_2_name,
             species=species_1, n_obs=n_obs_1)

for cur_model_name, cur_pred_fun in pred_funs_2.items():

    print(f'Predicting species 2 for {cur_model_name}')

    cur_x, cur_y, cur_z = conditional_plot_2d(
        cur_pred_fun, [cov_1, cov_2], n_covs, n_points=100)

    np.savez(join(target_dir, f'{cur_model_name}_2'),
             x=cur_x, y=cur_y, z=cur_z, model_name=cur_model_name,
             cov_1_name=cov_1_name, cov_2_name=cov_2_name,
             species=species_2, n_obs=n_obs_2)
