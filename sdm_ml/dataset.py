import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def get_training_set(self):
        pass

    @abstractmethod
    def get_test_set(self):
        pass


class BBSDataset(Dataset):

    def __init__(self, csv_folder, bio_covariates_only=True):

        self.csv_folder = csv_folder

        self.covariates = self.read_from_folder('x')

        if bio_covariates_only:
            # Remove non-bio covariates
            of_interest = [x for x in self.covariates.columns if 'bio' in x]
            self.covariates = self.covariates[of_interest]

        self.outcomes = self.read_from_folder('route.presence.absence')

        self.in_train = self.read_from_folder('in.train')
        self.in_train = np.squeeze(self.in_train.values)

        self.lat_lon = self.read_from_folder('latlon')

    @staticmethod
    def read_csv_iso_encoded(csv):

        return pd.read_csv(csv, index_col=0, encoding="ISO-8859-1")

    def read_from_folder(self, csv_name):

        return BBSDataset.read_csv_iso_encoded(
            os.path.join(self.csv_folder, csv_name + '.csv'))

    def get_lat_lon(self):

        return {'train': self.lat_lon.loc[self.in_train],
                'test': self.lat_lon.loc[~self.in_train]}

    def get_training_set(self):

        train_cov = self.covariates.loc[self.in_train, :]
        train_outcomes = self.outcomes.loc[self.in_train, :]

        return {'covariates': train_cov, 'outcomes': train_outcomes}

    def get_test_set(self):

        test_cov = self.covariates.loc[~self.in_train, :]
        test_outcomes = self.outcomes.loc[~self.in_train, :]

        return {'covariates': test_cov, 'outcomes': test_outcomes}
