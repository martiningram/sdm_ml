import os
import numpy as np
import pandas as pd
from typing import NamedTuple
from abc import ABC, abstractproperty


class Dataset(ABC):

    @abstractproperty
    def species_names(self):
        pass

    @abstractproperty
    def training_set(self):
        pass

    @abstractproperty
    def test_set(self):
        pass


class SpeciesData(NamedTuple):

    covariates: pd.DataFrame
    outcomes: pd.DataFrame
    lat_lon: pd.DataFrame = None


class BBSDataset(Dataset):

    def __init__(self, csv_folder, bio_covariates_only=True,
                 max_outcomes=None):

        self.csv_folder = csv_folder
        self.covariates = self.read_from_folder('x')
        self.species_info = self.read_from_folder('species.data')
        self.species_info.index = self.species_info.index.str.replace('_', ' ')

        if bio_covariates_only:
            # Remove non-bio covariates
            of_interest = [x for x in self.covariates.columns if 'bio' in x]
            self.covariates = self.covariates[of_interest]

        self.outcomes = self.read_from_folder('route.presence.absence')

        # Ensure we have species information for all of the species
        assert(all([x in self.species_info.index for x in
                    self.outcomes.columns]))

        if max_outcomes is not None:
            self.outcomes = self.outcomes[
                self.outcomes.columns[:max_outcomes]]

        self.in_train = self.read_from_folder('in.train')
        self.in_train = np.squeeze(self.in_train.values)
        self.lat_lon = self.read_from_folder('latlon')

        self.outcomes = self.outcomes.astype(int)
        self.covariates = self.covariates.astype(float)

    @classmethod
    def init_using_env_variable(cls):

        assert 'BBS_PATH' in os.environ
        return cls(os.environ['BBS_PATH'])

    @property
    def species_names(self):
        return self.outcomes.columns

    @staticmethod
    def read_csv_iso_encoded(csv):

        return pd.read_csv(csv, index_col=0, encoding="ISO-8859-1")

    def read_from_folder(self, csv_name):

        return BBSDataset.read_csv_iso_encoded(
            os.path.join(self.csv_folder, csv_name + '.csv'))

    @property
    def training_set(self):

        train_cov = self.covariates.loc[self.in_train, :]
        train_outcomes = self.outcomes.loc[self.in_train, :]
        train_lat_lon = self.lat_lon[self.in_train]

        return SpeciesData(covariates=train_cov,
                           outcomes=train_outcomes,
                           lat_lon=train_lat_lon)

    @property
    def test_set(self):

        test_cov = self.covariates.loc[~self.in_train, :]
        test_outcomes = self.outcomes.loc[~self.in_train, :]
        test_lat_lon = self.lat_lon[~self.in_train]

        return SpeciesData(covariates=test_cov,
                           outcomes=test_outcomes,
                           lat_lon=test_lat_lon)
