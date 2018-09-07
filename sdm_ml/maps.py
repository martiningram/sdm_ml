import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def produce_maps(dataset, fit_model, target_dir):

    # TODO: This will only work with the bbs dataset, because the dictionaries
    # are not enforced. Fix.

    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    # For now, pool all data
    lat_lon = dataset.get_lat_lon()
    train_set = dataset.get_training_set()
    test_set = dataset.get_test_set()

    features = pd.concat([train_set['covariates'], test_set['covariates']])

    # Predict
    predictions = fit_model.predict(features.values)
    predictions = pd.DataFrame(predictions, columns=dataset.species_names)

    # Plot
    both_lat_lon = pd.concat([lat_lon['train'], lat_lon['test']])

    x = both_lat_lon['Longitude']
    y = both_lat_lon['Latitude']

    f, ax = plt.subplots(1, 1)
    f.set_size_inches(8, 4)

    for cur_species in dataset.species_names:

        cur_predictions = predictions[cur_species]
        plot = ax.scatter(x, y, c=cur_predictions, vmin=0, vmax=1)
        cb = plt.colorbar(plot, ax=ax)
        ax.set_title(cur_species)
        f.tight_layout()
        plt.savefig(os.path.join(target_dir, cur_species + '.png'), dpi=300)
        ax.clear()
        cb.remove()
