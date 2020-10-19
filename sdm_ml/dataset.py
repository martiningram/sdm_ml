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
    # NB: This is the BBSDataset to go with the Dave Harris version.

    def __init__(self, csv_folder, bio_covariates_only=True, max_outcomes=None):

        self.csv_folder = csv_folder
        self.covariates = self.read_from_folder("x")
        self.species_info = self.read_from_folder("species.data")
        self.species_info.index = self.species_info.index.str.replace("_", " ")

        if bio_covariates_only:
            # Remove non-bio covariates
            of_interest = [x for x in self.covariates.columns if "bio" in x]
            self.covariates = self.covariates[of_interest]

        self.outcomes = self.read_from_folder("route.presence.absence")

        # Ensure we have species information for all of the species
        assert all([x in self.species_info.index for x in self.outcomes.columns])

        if max_outcomes is not None:
            self.outcomes = self.outcomes[self.outcomes.columns[:max_outcomes]]

        self.in_train = self.read_from_folder("in.train")
        self.in_train = np.squeeze(self.in_train.values)
        self.lat_lon = self.read_from_folder("latlon")

        self.outcomes = self.outcomes.astype(int)
        self.covariates = self.covariates.astype(float)

    @classmethod
    def init_using_env_variable(cls, **kwargs):

        assert "BBS_PATH" in os.environ
        return cls(os.environ["BBS_PATH"], **kwargs)

    @property
    def species_names(self):
        return self.outcomes.columns

    @staticmethod
    def read_csv_iso_encoded(csv):

        return pd.read_csv(csv, index_col=0, encoding="ISO-8859-1")

    def read_from_folder(self, csv_name):

        return BBSDataset.read_csv_iso_encoded(
            os.path.join(self.csv_folder, csv_name + ".csv")
        )

    @property
    def training_set(self):

        train_cov = self.covariates.loc[self.in_train, :]
        train_outcomes = self.outcomes.loc[self.in_train, :]
        train_lat_lon = self.lat_lon[self.in_train]

        return SpeciesData(
            covariates=train_cov, outcomes=train_outcomes, lat_lon=train_lat_lon
        )

    @property
    def test_set(self):

        test_cov = self.covariates.loc[~self.in_train, :]
        test_outcomes = self.outcomes.loc[~self.in_train, :]
        test_lat_lon = self.lat_lon[~self.in_train]

        return SpeciesData(
            covariates=test_cov, outcomes=test_outcomes, lat_lon=test_lat_lon
        )


class BBS2019(Dataset):
    def __init__(self, bbs_csv, covariates_csv, train_folds=[1, 2, 3], test_folds=[4]):

        self.dataset = load_bbs_dataset_2019(
            bbs_csv, covariates_csv, train_folds, test_folds
        )

    @property
    def species_names(self):
        return self.dataset["train"].outcomes.columns

    @property
    def training_set(self):
        return self.dataset["train"]

    @property
    def test_set(self):
        return self.dataset["test"]


def load_bbs_dataset_2019(
    bbs_csv: str, covariates_csv: str, train_folds=[1, 2, 3], test_folds=[4]
):

    # NB: This loads the BBS dataset in its 2019 iteration.
    pa_df = pd.read_csv(bbs_csv, index_col=0)
    covariates = pd.read_csv(covariates_csv, index_col=0)

    pa_cells = pa_df["cell_id"].values
    covariates = covariates.set_index("cell")

    pa_covs = covariates.loc[pa_cells]
    is_train = pa_df["fold_id"].isin(train_folds)
    lat_lon = pa_df[["X", "Y"]]

    outcome_cols = [
        x for x in pa_df.columns if x not in ["X", "Y", "fold_id", "cell_id"]
    ]

    return {
        "train": SpeciesData(
            covariates=pa_covs[is_train.values],
            outcomes=pa_df[is_train.values][outcome_cols],
            lat_lon=lat_lon[is_train.values],
        ),
        "test": SpeciesData(
            covariates=pa_covs[~is_train.values],
            outcomes=pa_df[~is_train.values][outcome_cols],
            lat_lon=lat_lon[~is_train.values],
        ),
    }
