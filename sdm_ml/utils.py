import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .dataset import BBSDataset
import pandas as pd
from tqdm import tqdm


def get_bbs_dataset(n_subset_sites=None, n_subset_species=None, seed=2):
    # A helper function to avoid me having to import and prepare the dataset
    # every time

    np.random.seed(seed)

    dataset = BBSDataset.init_using_env_variable()

    X = dataset.training_set.covariates.values
    y = dataset.training_set.outcomes.values

    if n_subset_sites is not None:
        chosen_sites = np.random.choice(X.shape[0], size=n_subset_sites, replace=False)
    else:
        chosen_sites = np.arange(X.shape[0])

    if n_subset_species is not None:
        chosen_species = np.random.choice(
            y.shape[1], size=n_subset_species, replace=False
        )
    else:
        chosen_species = np.arange(y.shape[1])

    chosen_species_names = dataset.outcomes.columns[chosen_species]

    X = X[chosen_sites, :]
    y = y[chosen_sites][:, chosen_species]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, y, chosen_species_names, chosen_sites, scaler


def plot_maps(model, covariates, species_names, longitude, latitude, target_dir):

    os.makedirs(target_dir, exist_ok=True)

    y_p = model.predict_marginal_probabilities(covariates)
    y_p = pd.DataFrame(y_p, columns=species_names)

    for cur_species in tqdm(y_p.columns):

        cur_y_p = y_p[cur_species]
        save_as = cur_species.replace("/", "or")

        f, ax = plt.subplots(1, 1)
        mappable = ax.scatter(longitude, latitude, c=cur_y_p, s=1.0)
        plt.colorbar(mappable, ax=ax)
        ax.set_title(cur_species)

        f.set_size_inches(8, 4)
        f.tight_layout()

        plt.savefig(os.path.join(target_dir, f"{save_as}.png"), dpi=300)

        plt.close(f)
