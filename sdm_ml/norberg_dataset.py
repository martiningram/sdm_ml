import os
import pandas as pd
from os.path import join
from .dataset import Dataset, SpeciesData
from functools import partial


class NorbergDataset(Dataset):

    def __init__(self, base_dir, dataset_name, cv_fold):

        # Make sure arguments make sense
        assert dataset_name in ['birds', 'plant', 'vegetation', 'trees',
                                'butterfly']
        assert cv_fold in [1, 2, 3]

        suffix = f'{cv_fold}_{dataset_name}'

        xt_file = join(base_dir, f'Xt_{suffix}.csv')
        yt_file = join(base_dir, f'Yt_{suffix}.csv')
        st_file = join(base_dir, f'St_{suffix}.csv')
        xv_file = join(base_dir, f'Xv_{suffix}.csv')
        yv_file = join(base_dir, f'Yv_{suffix}.csv')
        sv_file = join(base_dir, f'Sv_{suffix}.csv')

        load_no_header = partial(pd.read_csv, header=None)

        # TODO: Could make up some names for these covariates
        self.train_covariates = load_no_header(xt_file)

        # TODO: I need to find the species names
        self.train_outcomes = load_no_header(yt_file)

        # TODO: I should probably enforce that this has lat and lon
        # Not returning for now
        self.train_locations = load_no_header(st_file)

        # Same for test (or validation)
        self.test_covariates, self.test_outcomes, self.test_locations = [
            load_no_header(x) for x in [xv_file, yv_file, sv_file]]

    @classmethod
    def init_using_env_variable(cls, dataset_name, cv_fold, **kwargs):

        assert 'NORBERG_PATH' in os.environ

        return cls(os.environ['NORBERG_PATH'], dataset_name, cv_fold, **kwargs)

    @property
    def training_set(self):

        return SpeciesData(covariates=self.train_covariates,
                           outcomes=self.train_outcomes)

    @property
    def test_set(self):

        return SpeciesData(covariates=self.test_covariates,
                           outcomes=self.test_outcomes)

    @property
    def species_names(self):

        return self.train_outcomes.columns
