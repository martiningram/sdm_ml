import sys
import pandas as pd
from os import makedirs
from os.path import join
from sdm_ml.dataset import BBSDataset


train_test_split = sys.argv[1]
target_dir = sys.argv[2]
min_count = int(sys.argv[3])

makedirs(target_dir, exist_ok=True)

dataset = BBSDataset.init_using_env_variable()

full_covs = dataset.training_set.covariates
full_outcomes = dataset.training_set.outcomes

is_train = pd.read_csv(train_test_split)['x'].values

train_covs = full_covs[is_train]
train_outcomes = full_outcomes[is_train]

test_covs = full_covs[~is_train]
test_outcomes = full_outcomes[~is_train]

counts_train = train_outcomes.sum(axis=0)
counts_test = test_outcomes.sum(axis=0)

print(f'Originally had {counts_train.shape[0]} species.')

to_keep = counts_train[
    (counts_train >= min_count) & (counts_test >= min_count)].index

print(f'Keeping {to_keep.shape[0]} species.')

train_outcomes = train_outcomes[to_keep]
test_outcomes = test_outcomes[to_keep]

train_covs.to_csv(join(target_dir, 'train_covariates.csv'))
train_outcomes.to_csv(join(target_dir, 'train_outcomes.csv'))

test_covs.to_csv(join(target_dir, 'test_covariates.csv'))
test_outcomes.to_csv(join(target_dir, 'test_outcomes.csv'))
