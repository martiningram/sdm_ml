import os
import numpy as np
import pandas as pd
from typing import NamedTuple
from abc import ABC, abstractproperty
from ml_tools.modelling import remove_correlated_variables


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
    bbs_csv: str,
    covariates_csv: str,
    train_folds=[1, 2, 3],
    test_folds=[4],
    min_train_occurrences=2,
    drop_correlated_covariates=True,
):

    # NB: This loads the BBS dataset in its 2019 iteration.
    pa_df = pd.read_csv(bbs_csv, index_col=0)
    covariates = pd.read_csv(covariates_csv, index_col=0)

    pa_cells = pa_df["cell_id"].values
    covariates = covariates.set_index("cell")
    covariates = covariates[[x for x in covariates.columns if x not in ["x", "y"]]]

    pa_covs = covariates.loc[pa_cells]
    is_train = pa_df["fold_id"].isin(train_folds)
    lat_lon = pa_df[["X", "Y"]]

    if drop_correlated_covariates:
        train_covs = pa_covs[is_train.values]
        covs_to_keep = remove_correlated_variables(train_covs)
        pa_covs = pa_covs[covs_to_keep]

    outcome_cols = [
        x for x in pa_df.columns if x not in ["X", "Y", "fold_id", "cell_id"]
    ]

    # TODO: Also drop correlated covariates

    # Remove outcome columns with slashes that were not assigned uniquely, or
    # just assigned to "sp."
    outcome_cols = [x for x in outcome_cols if "/" not in x and "sp." not in x]

    # Collapse the species with subspecies
    n_words = pd.Series([len(x.split(" ")) for x in outcome_cols], index=outcome_cols,)
    to_collapse = n_words[n_words > 2].index

    outcomes = pa_df[outcome_cols]
    new_outcomes = outcomes.copy()

    for cur_species in to_collapse:

        target = " ".join(cur_species.split(" ")[:2])

        if target not in new_outcomes.columns:
            new_outcomes[target] = np.repeat(False, new_outcomes.shape[0])

        new_outcomes[target] = new_outcomes[target] | new_outcomes[cur_species]

    new_outcomes = new_outcomes.drop(columns=to_collapse)

    n_train_obs = new_outcomes[is_train.values].sum()
    to_keep = n_train_obs[n_train_obs > min_train_occurrences].index

    return {
        "train": SpeciesData(
            covariates=pa_covs[is_train.values],
            outcomes=new_outcomes[is_train.values][to_keep],
            lat_lon=lat_lon[is_train.values],
        ),
        "test": SpeciesData(
            covariates=pa_covs[~is_train.values],
            outcomes=new_outcomes[~is_train.values][to_keep],
            lat_lon=lat_lon[~is_train.values],
        ),
    }
