from sklearn.cluster import MiniBatchKMeans


def find_starting_z(X, num_inducing):
    # TODO: Maybe move this to GP utils or something
    # Find starting locations for inducing points
    k_means = MiniBatchKMeans(n_clusters=num_inducing)
    k_means.fit(X)
    Z = k_means.cluster_centers_
    return Z
