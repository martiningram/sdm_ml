import sys
import pickle
import numpy as np
import pandas as pd
import json
from os import makedirs
from os.path import join
from svgp.tf.models.mogp_classifier import fit, save_results
from sklearn.preprocessing import StandardScaler
from time import time


train_covariates = pd.read_csv(sys.argv[1], index_col=0)
train_outcomes = pd.read_csv(sys.argv[2], index_col=0)
target_dir = sys.argv[3]
n_inducing = int(sys.argv[4])
n_latent = int(sys.argv[5])
seed = int(sys.argv[6])
is_test_run = int(sys.argv[7]) == 1

scaler = StandardScaler()

X = scaler.fit_transform(train_covariates.values)
y = train_outcomes.values

start_time = time()
mogp_result = fit(X.astype(np.float32), y.astype(np.float32), n_inducing,
                  n_latent, random_seed=seed, test_run=is_test_run)
runtime = time() - start_time

makedirs(target_dir, exist_ok=True)

save_results(mogp_result, join(target_dir, 'results.npz'))

# Save settings
settings = {
    'n_latent': n_latent,
    'n_inducing': n_inducing,
    'seed': seed,
    'is_test_run': is_test_run,
    'runtime': runtime
}

with open(join(target_dir, 'settings.json'), 'w') as f:
    json.dump(settings, f)

with open(join(target_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
