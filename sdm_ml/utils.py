import numpy as np
from sklearn.preprocessing import StandardScaler
from .dataset import BBSDataset


def get_bbs_dataset(n_subset_sites=None, n_subset_species=None, seed=2):
    # A helper function to avoid me having to import and prepare the dataset
    # every time

    np.random.seed(seed)

    dataset = BBSDataset.init_using_env_variable()

    X = dataset.training_set.covariates.values
    y = dataset.training_set.outcomes.values

    if n_subset_sites is not None:
        chosen_sites = np.random.choice(X.shape[0], size=n_subset_sites,
                                        replace=False)
    else:
        chosen_sites = np.arange(X.shape[0])

    if n_subset_species is not None:
        chosen_species = np.random.choice(y.shape[1], size=n_subset_species,
                                          replace=False)
    else:
        chosen_species = np.arange(y.shape[1])

    chosen_species_names = dataset.outcomes.columns[chosen_species]

    X = X[chosen_sites, :]
    y = y[chosen_sites][:, chosen_species]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, y, chosen_species_names, chosen_sites, scaler
