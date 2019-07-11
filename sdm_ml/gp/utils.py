from sklearn.cluster import MiniBatchKMeans, KMeans


def find_starting_z(X, num_inducing, use_minibatching=True):
    # Find starting locations for inducing points

    if use_minibatching:
        k_means = MiniBatchKMeans(n_clusters=num_inducing)
    else:
        k_means = KMeans(n_clusters=num_inducing)

    k_means.fit(X)
    Z = k_means.cluster_centers_

    return Z
