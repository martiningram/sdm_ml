import numpy as np
from abc import ABC, abstractmethod


class PresenceAbsenceModel(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """ Fits a PresenceAbsenceModel.

        Args:
            X: An NxM matrix, with N the number of training rows,
               and M the number of covariates.
            y: A binary NxK matrix, with N the number of training
               rows and K the number of species considered. Entry [i, j] is 1
               if species j was observed at site i, and zero otherwise.

        Returns:
            Nothing, but fits the PresenceAbsenceModel.
        """
        pass

    @abstractmethod
    def predict_log_marginal_probabilities(self, X: np.ndarray) -> np.ndarray:
        """ Predicts marginal probabilities using the PresenceAbsenceModel.

        Args:
            X: An NxM matrix, with N the number of prediction examples and M
               the number of covariates.

        Returns:
            A numpy array of shape [N, K, 2] filled with marginal log
            probabilities of absence and presence for each of the K species.
        """
        pass

    @abstractmethod
    def calculate_log_likelihood(self, X: np.ndarray, y: np.ndarray) \
            -> np.ndarray:
        """ Calculates the log likelihood of the outcomes y given covariates X.

        Args:
            X: An NxM matrix with N the number of prediction examples and M
               the number of covariates.
            y: A binary NxK matrix, with N the number of training rows and K
               the number of species considered. Entry [i, j] is 1 if species
               j was observed at site i, and zero otherwise.

        Returns:
            A vector of length N containing the log likelihood of each site's
            observations under the model.
        """
        pass

    @abstractmethod
    def save_model(self, target_folder: str) -> None:
        """ Saves the model to files in the target folder.

        Args:
            target_file: Where to save the model parameters [folder name].

        Returns:
            Nothing, but stores model parameters in files in a folder at the
            path given.
        """
        pass

    def predict_marginal_probabilities(self, X: np.ndarray) -> np.ndarray:
        """A convenience function which predicts the marginal probabilities
        of presence (not on the log scale).

        Args:
            X: The data to predict in an NxM matrix with N sites, M covariates.

        Returns:
            An array of size NxK, where K is the number of species.
        """

        log_probs = self.predict_log_marginal_probabilities(X)
        presence_probs = np.exp(log_probs[..., 1])
        return presence_probs
