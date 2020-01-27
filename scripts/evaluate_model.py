import os
import time
import numpy as np
from os.path import join
from functools import partial

from sdm_ml.dataset import BBSDataset, SpeciesData
from sdm_ml.norberg_dataset import NorbergDataset
from sdm_ml.scikit_model import ScikitModel
from sdm_ml.evaluation import compute_and_save_results_for_evaluation
from ml_tools.utils import create_path_with_variables
from sdm_ml.base_rate_model import BaseRateModel
from sklearn.linear_model import LogisticRegression


def evaluate_model(training_set, test_set, model, output_dir):

    np.save(join(output_dir, 'names'), test_set.outcomes.columns.values)

    start_time = time.time()
    model.fit(training_set.covariates.values,
              training_set.outcomes.values.astype(int))
    end_time = time.time()
    time_taken = end_time - start_time

    with open(join(output_dir, 'runtime.txt'), 'w') as f:
        f.write(str(time_taken))

    compute_and_save_results_for_evaluation(test_set, model, output_dir)


def get_mixed_stan(n_dims, n_outcomes):

    from sdm_ml.hierarchical.independent_hierarchical_model import \
        IndependentHierarchicalModel

    return IndependentHierarchicalModel()


def get_hierarchical_mogp(n_dims, n_outcomes, n_inducing, n_latent, kernel):

    from sdm_ml.gp.hierarchical_mogp import HierarchicalMOGP

    model = HierarchicalMOGP(n_inducing, n_latent, kernel)

    return model


def get_new_sogp(n_dims, n_outcomes, n_inducing, kernel):

    from sdm_ml.gp.sogp import SOGP

    model = SOGP(n_inducing, kernel)

    return model


def get_brt(n_dims, n_outcomes):

    from sdm_ml.brt.dismo_brt import DismoBRT

    return ScikitModel(DismoBRT)


def shuffle_train_set_order(training_set, seed=1):

    np.random.seed(seed)
    n_sites = training_set.covariates.shape[0]
    new_order = np.random.choice(n_sites, size=n_sites, replace=False)

    lat_lon = (None if training_set.lat_lon is None else
               training_set.lat_lon.iloc[new_order])

    return SpeciesData(
        covariates=training_set.covariates.iloc[new_order],
        outcomes=training_set.outcomes.iloc[new_order],
        lat_lon=lat_lon
    )


def discard_rare_species(training_set, test_set, min_presences=5):

    train_outcomes = training_set.outcomes
    n_presences = train_outcomes.sum(axis=0)
    make_cut = train_outcomes.columns[n_presences > min_presences]

    new_training_set = SpeciesData(
        covariates=training_set.covariates,
        outcomes=training_set.outcomes[make_cut],
        lat_lon=training_set.lat_lon
    )

    new_test_set = SpeciesData(
        covariates=test_set.covariates,
        outcomes=test_set.outcomes[make_cut],
        lat_lon=test_set.lat_lon
    )

    return new_training_set, new_test_set


def get_log_reg(n_dims, n_outcomes):

    model = ScikitModel()

    return model


def get_log_reg_unregularised(n_dims, n_outcomes):

    model = ScikitModel(partial(LogisticRegression, penalty='none',
                                solver='newton-cg'))

    return model


def get_base_rate_model(n_dims, n_outcomes):

    return ScikitModel(BaseRateModel)


def get_random_forest_cv(n_dims, n_outcomes):

    model = ScikitModel(partial(ScikitModel.create_cross_validated_forest,
                                n_covariates=n_dims))

    return model


def pick_random_species_and_sites(n_sites_to_pick, n_species_to_pick,
                                  n_sites, n_species):

    assert n_sites_to_pick < n_sites and n_species_to_pick < n_species

    picked_sites = np.random.choice(n_sites, size=n_sites_to_pick,
                                    replace=False)
    picked_species = np.random.choice(n_species, size=n_species_to_pick,
                                      replace=False)

    return picked_sites, picked_species


def reduce_sites(species_data, picked_sites):

    new_lat_lon = (species_data.lat_lon if species_data.lat_lon is None
                   else species_data.lat_lon.iloc[picked_sites])

    return SpeciesData(
        covariates=species_data.covariates.iloc[picked_sites],
        outcomes=species_data.outcomes.iloc[picked_sites],
        lat_lon=new_lat_lon)


def reduce_species(species_data, picked_species):

    return SpeciesData(
        covariates=species_data.covariates,
        outcomes=species_data.outcomes.iloc[:, picked_species],
        lat_lon=species_data.lat_lon)


if __name__ == '__main__':

    test_run = False
    output_base_dir = os.environ['SDM_ML_EVAL_PATH']
    min_presences = 5

    datasets = {}
    datasets = NorbergDataset.fetch_all_norberg_sets()
    datasets = {x: y for x, y in datasets.items() if '3' not in x}
    datasets['bbs'] = BBSDataset.init_using_env_variable()

    target_dir = join(output_base_dir,
                      create_path_with_variables(test_run=test_run))

    models = {
        # 'brt': get_brt,
        # 'rf_cv': get_random_forest_cv,
        # 'log_reg_unreg': get_log_reg_unregularised,
        'hierarchical_mogp_10': partial(
            get_hierarchical_mogp, n_inducing=100, n_latent=10,
            kernel='matern_3/2'),
        'sogp': partial(get_new_sogp, n_inducing=100, kernel='matern_3/2')
    }

    for cur_dataset_name, cur_dataset in datasets.items():

        training_set = cur_dataset.training_set
        test_set = cur_dataset.test_set

        training_set, test_set = discard_rare_species(
            training_set, test_set, min_presences=min_presences)

        training_set = shuffle_train_set_order(training_set)

        n_dims = training_set.covariates.shape[1]
        n_outcomes = training_set.outcomes.shape[1]

        if test_run:

            n_sites = training_set.covariates.shape[0]

            sites_to_pick = 100
            species_to_pick = 2

            picked_sites, picked_species = pick_random_species_and_sites(
                sites_to_pick, species_to_pick, n_sites, n_outcomes)

            training_set = reduce_sites(training_set, picked_sites)
            training_set = reduce_species(training_set, picked_species)
            test_set = reduce_species(test_set, picked_species)

            n_outcomes = species_to_pick

        subdir = join(target_dir, cur_dataset_name)

        for cur_model_name, cur_model_fn in models.items():

            cur_subdir = join(subdir, cur_model_name)
            os.makedirs(cur_subdir, exist_ok=True)

            cur_model = cur_model_fn(n_dims, n_outcomes)

            # try:
            evaluate_model(training_set, test_set, cur_model, cur_subdir)
            # except ValueError as e:
            #     print(f'Failed to fit {cur_model_name}. Error was: {e}')
            #     target_file = join(cur_subdir, 'error.txt')
            #     with open(target_file, 'w') as f:
            #         f.write(f'Failed to fit model. Error was: {e}')
            #     continue
