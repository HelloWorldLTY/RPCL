import numpy as np
from sklearn.datasets import make_blobs

def rpcl_kmeans(X, n_clusters, max_iter=100, penalty=0.1, tol=1e-4):
    """
    Rival Penalized Competitive Learning (RPCL) applied to K-means.

    Parameters:
        X (ndarray): Input data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters.
        max_iter (int): Maximum number of iterations.
        penalty (float): Rival penalty term.
        tol (float): Convergence tolerance.

    Returns:
        labels (ndarray): Cluster labels for each data point.
        centroids (ndarray): Final cluster centroids.
    """
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly from the data
    rng = np.random.default_rng()
    centroids = X[rng.choice(n_samples, n_clusters, replace=False)]

    for iteration in range(max_iter):
        # Compute distances between points and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # Find the closest and second-closest centroids for each point
        closest = np.argsort(distances, axis=1)
        nearest = closest[:, 0]  # Closest centroid
        second_nearest = closest[:, 1]  # Second closest (rival)

        # Update centroids
        new_centroids = np.copy(centroids)
        for k in range(n_clusters):
            # Points belonging to the current cluster
            mask = nearest == k

            # Rival penalty for second closest points
            rival_mask = second_nearest == k

            if np.sum(mask) > 0:
                new_centroids[k] += (X[mask].mean(axis=0) - centroids[k])

            if np.sum(rival_mask) > 0:
                new_centroids[k] -= penalty * (X[rival_mask].mean(axis=0) - centroids[k])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    # Assign labels based on closest centroids
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    return labels, centroids
