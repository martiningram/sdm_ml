import numpy as np
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans


def find_starting_z(X, num_inducing):
    # Find starting locations for inducing points
    k_means = MiniBatchKMeans(n_clusters=num_inducing)
    k_means.fit(X)
    Z = k_means.cluster_centers_
    return Z


def predict_with_link(means, variances, link_fun=norm.cdf, n_samples=4000):
    # Draws samples and apply link function.
    # Means: 1D array of means
    # variances: 1D array of variances (Note: variance, not standard deviation).
    sds = np.sqrt(variances)
    draws = norm.rvs(means, sds, size=(n_samples, means.shape[0]))
    return link_fun(draws)


def predict_and_summarise(means, variances, link_fun=norm.cdf, n_samples=4000,
                          chunk_into=1000):

    pred_means = list()

    num_pts = means.shape[0]
    num_chunks = (num_pts // chunk_into) + 1

    for cur_chunk in range(num_chunks):

        cur_start = cur_chunk * chunk_into
        cur_end = (cur_chunk + 1) * chunk_into

        cur_means = means[cur_start:cur_end]
        cur_vars = variances[cur_start:cur_end]

        cur_predictions = predict_with_link(
            cur_means, cur_vars, link_fun=link_fun, n_samples=n_samples)

        cur_means = np.mean(cur_predictions, axis=0)
        pred_means.append(cur_means)

    return np.concatenate(pred_means)
