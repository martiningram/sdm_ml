import sys
import pickle
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join
from svgp.tf.models.mogp_classifier import fit, save_results
from sklearn.preprocessing import StandardScaler


train_covariates = pd.read_csv(sys.argv[1])
train_outcomes = pd.read_csv(sys.argv[2])
target_dir = sys.argv[3]
n_inducing = int(sys.argv[4])
n_latent = int(sys.argv[5])
seed = int(sys.argv[6])

scaler = StandardScaler()

X = scaler.fit_transform(train_covariates.values)
y = train_outcomes.values

mogp_result = fit(X.astype(np.float32), y.astype(np.float32), n_inducing,
                  n_latent, random_seed=seed)

makedirs(target_dir, exist_ok=True)

save_results(join(target_dir, 'results.npz'))

with open(join(target_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
