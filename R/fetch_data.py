import os
from tqdm import tqdm
from os.path import join

from sdm_ml.dataset import BBSDataset
from sdm_ml.norberg_dataset import NorbergDataset


target_dir = './datasets/'

bbs_set = BBSDataset.init_using_env_variable()
norberg_set = NorbergDataset.fetch_all_norberg_sets()

all_sets = norberg_set
all_sets['bbs'] = bbs_set

for cur_name, cur_set in tqdm(all_sets.items()):

    cur_dir = join(target_dir, cur_name)

    os.makedirs(cur_dir, exist_ok=True)

    # Training set
    cur_set.training_set.covariates.to_csv(join(cur_dir, 'x_train.csv'))
    cur_set.training_set.outcomes.to_csv(join(cur_dir, 'y_train.csv'))

    # Test set
    cur_set.test_set.covariates.to_csv(join(cur_dir, 'x_test.csv'))
    cur_set.test_set.outcomes.to_csv(join(cur_dir, 'y_test.csv'))
