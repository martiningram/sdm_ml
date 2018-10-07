import numpy as np
from scipy.special import expit
from sparse_gp.kernels.rbf_kernel import RBFKernel
from sparse_gp.likelihoods.bernoulli_logit_likelihood import \
    BernoulliLogitLikelihood
from sparse_gp.inference.laplace_inference import LaplaceInference
from sklearn.preprocessing import StandardScaler
from sdm_ml.model import PresenceAbsenceModel
from sdm_ml.gp.utils import predict_and_summarise


class SingleOutputGPLaplace(PresenceAbsenceModel):

    def __init__(self, verbose=False, n_draws_pred=4000):

        self.scaler = None
        self.verbose = verbose
        self.n_draws_pred = n_draws_pred

    def fit(self, X, y):

        self.models = list()
        self.scaler = StandardScaler()

        X = self.scaler.fit_transform(X)

        for i in tqdm(range(y.shape[1])):

            cur_outcomes = y[:, i].astype(float)
            cur_kernel = RBFKernel(np.arange(X.shape[1]))
            cur_likelihood = BernoulliLogitLikelihood()
            cur_inference = LaplaceInference(cur_kernel, cur_likelihood,
                                             verbose=self.verbose)

            # Fit the model
            cur_inference.fit(X, cur_outcomes)

            if self.verbose:
                print(cur_inference.kernel)

            self.models.append(cur_inference)

    def predict(self, X):

        assert(len(self.models) > 0)
        predictions = list()

        X = self.scaler.transform(X)

        for m in self.models:

            means, variances = m.predict(X)

            pred_mean_prob = predict_and_summarise(
                means, variances, link_fun=expit, n_samples=self.n_draws_pred)

            predictions.append(pred_mean_prob)

        return np.stack(predictions, axis=1)

    def save_parameters(self, target_folder):

        self.create_folder(target_folder)

        # TODO: Maybe save something with names instead...?
        flat_params = np.array([x.kernel.get_flat_hyperparameters() for x in
                                self.models])

        np.save(os.path.join(target_folder, 'flat_hyperparams.npy'),
                flat_params)
