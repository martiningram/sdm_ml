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
    # TODO: This creates a big array of samples and then takes their mean. If
    # "means" is very large, it may be better to do this in stages.
    sds = np.sqrt(variances)
    draws = norm.rvs(means, sds, size=(n_samples, means.shape[0]))
    return link_fun(draws)
