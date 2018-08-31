from abc import ABC, abstractmethod


class PresenceAbsenceModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        """ Fits a PresenceAbsenceModel.

        Args:
            X (np.array): An NxM matrix, with N the number of training rows, and
                M the number of covariates.
            y (np.array): A binary NxK matrix, with N the number of training
            rows and K the number of species considered. Entry [i, j] is 1 if
            species j was observed at site i, and zero otherwise.

        Returns:
            Nothing, but fits the PresenceAbsenceModel.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """ Predicts using the PresenceAbsenceModel.

        Args:
            X (np.array): An NxM matrix, with N the number of prediction
                examples and M the number of covariates.

        Returns:
            np.array: A numpy array of shape [N, K] filled with probabilities
                of presence for each of the K species.
        """
        pass
