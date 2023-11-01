import numpy as np
from modules.distances import get_distance_matrix

# Davies-Bouldin index
def internal_indices(X, G, norma, cov_i, centroids):
    
    D_Centroids=get_distance_matrix(centroids,centroids,cov_i,norms=[norma]).squeeze()
    D_CentroidtoX=get_distance_matrix(centroids,X,cov_i,norms=[norma]).squeeze()
    avrgD_CentroidToX = np.ma.masked_equal(np.multiply(D_CentroidtoX, G.T), 0.0, copy=False).mean(axis=1)
    minD_CentroidToX = np.ma.masked_equal(np.multiply(D_CentroidtoX, G.T), 0.0, copy=False).min(axis=1)

    """ Davies-Bouldin index. This index is a measure of how well a clustering algorithm
    separates the clusters. The lower the index, the better the clustering. The idea behind this index
    is to measure the distance between clusters and the distance between points within each cluster. It does so
    by calculating the average distance between each cluster and its points and the distance between each pair of clusters.
    This index is a measure of cohesion and separation, which means that it is a measure of how well the clusters are
    separated and how well the points are clustered within each cluster.
    """ 
    # davies-bouldin index
    db=0
    for c in range(centroids.shape[0]):
        D_c = D_Centroids[c,:]
        db += np.divide((avrgD_CentroidToX+avrgD_CentroidToX[c]), D_c, out=np.zeros_like(D_c), where=D_c!=0).max()
    yield db / centroids.shape[0]

    """ Dunn index. This index is a measure of how well a clustering algorithm separates the clusters.
    The higher the index, the better the clustering. The idea behind this index is to measure the distance
    between clusters and the distance between points within each cluster. It does so by calculating the minimum
    distance between each cluster and its points and the maximum distance between each pair of clusters.
    This index is a measure of cohesion and separation, which means that it is a measure of how well the clusters are
    separated and how well the points are clustered within each cluster.
    """
    # dunn index
    yield minD_CentroidToX.min() / D_Centroids.max()

    """ Calinski-Harabasz index. This index is a measure of how well a clustering algorithm separates the clusters.
    The higher the index, the better the clustering. The idea behind this index is to measure the distance between
    clusters and the distance between points within each cluster. It does so by calculating the average distance between
    each cluster and its points and the distance between each pair of clusters. This index is a measure of cohesion and
    separation, which means that it is a measure of how well the clusters are separated and how well the points are
    clustered within each cluster.
    """
    # calinski-harabasz index
    avrgD_XToavrgX = get_distance_matrix(X,[X.mean(axis=0)],cov_i,norms=[norma]).mean()
    yield ((G.shape[1] * avrgD_CentroidToX - avrgD_XToavrgX)**2).sum() / (G.shape[1]  * D_Centroids.sum())