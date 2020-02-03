import pickle
import pandas as pd
from os import makedirs
from os.path import join
from sdm_ml.dataset import BBSDataset
from sdm_ml.norberg_dataset import NorbergDataset
from sklearn.preprocessing import StandardScaler


bbs_dataset = BBSDataset.init_using_env_variable()
datasets = NorbergDataset.fetch_all_norberg_sets(cv_folds=[1])
datasets['bbs'] = bbs_dataset

target_base_dir = './datasets'
makedirs(target_base_dir, exist_ok=True)

for cur_dataset_name, cur_dataset in datasets.items():

    # Create directory:
    cur_target_dir = join(target_base_dir, cur_dataset_name)
    makedirs(cur_target_dir, exist_ok=True)

    # Prepare data
    scaler = StandardScaler()
    cur_covariates = cur_dataset.training_set.covariates

    X = scaler.fit_transform(cur_covariates.values)
    X_test = scaler.transform(cur_dataset.test_set.covariates.values)

    Y = cur_dataset.training_set.outcomes
    Y_test = cur_dataset.test_set.outcomes

    # Store data
    pd.DataFrame(X).to_csv(join(cur_target_dir, 'X.csv'))
    pd.DataFrame(X_test).to_csv(join(cur_target_dir, 'X_test.csv'))

    Y.to_csv(join(cur_target_dir, 'Y.csv'))
    Y_test.to_csv(join(cur_target_dir, 'Y_test.csv'))

    with open(join(cur_target_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
