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
    # TODO: This could create a big array (unnecessary really). If N gets
    # really large, consider doing this in stages.
    # Draw samples and apply link function
    sds = np.sqrt(variances)
    draws = norm.rvs(means, sds, size=(n_samples, means.shape[0]))
    return link_fun(draws)
