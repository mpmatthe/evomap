"""
Module for transforming lower-dimensional maps post-creation, including alignment and rotation.
"""

from scipy.linalg import orthogonal_procrustes
import numpy as np

def align_maps(Xs, X_ref):
    """
    Align a sequence of maps to a reference map using Orthogonal Procrustes Analysis.

    Parameters
    ----------
    Xs : list of ndarray
        List of map coordinates, each of shape (n_samples, n_dims)
    X_ref : ndarray
        Reference map, shape (n_samples, n_dims)

    Returns
    -------
    list of ndarray
        List of aligned map coordinates, each of shape (n_samples, n_dims)
    """
    if not Xs or X_ref.size == 0:
        raise ValueError("Input maps and reference map must not be empty.")
    for X in Xs:
        if X.shape[1] != X_ref.shape[1]:
            raise ValueError("All maps must have the same number of dimensions as the reference map.")

    R, _ = orthogonal_procrustes(Xs[0], X_ref)
    return [X @ R for X in Xs]

def align_map(X, X_ref):
    """
    Align a single map to a reference map using Orthogonal Procrustes Analysis.

    Parameters
    ----------
    X : ndarray
        Map coordinates, shape (n_samples, n_dims)
    X_ref : ndarray
        Reference map, shape (n_samples, n_dims)

    Returns
    -------
    ndarray
        Aligned map, shape (n_samples, n_dims)
    """
    if X.size == 0 or X_ref.size == 0:
        raise ValueError("Input maps must not be empty.")
    if X.shape != X_ref.shape:
        raise ValueError("Input map and reference map must have the same shape.")    
    X_orig = X.copy()
    R, _ = orthogonal_procrustes(X_orig, X_ref)
    return np.dot(X_orig, R)

def PCA(X, num_components):
    """
    Perform Principal Component Analysis (PCA).

    Parameters
    ----------
    X : ndarray
        Data matrix, shape (n_samples, n_features)
    num_components : int
        Number of principal components to retain

    Returns
    -------
    ndarray
        Reduced dimensionality data, shape (n_samples, num_components)
    """
    X_meaned = X - np.mean(X, axis=0)
    cov_mat = np.cov(X_meaned, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigenvector_subset = eigen_vectors[:, sorted_indices[:num_components]]
    return np.dot(X_meaned, eigenvector_subset)

def rotate_map(Y_2D):
    """
    Rotate a 2D map to align along the direction of maximum variance using PCA.

    Parameters
    ----------
    Y_2D : ndarray
        2D map, shape (n_samples, 2)

    Returns
    -------
    ndarray
        Rotated map, shape (n_samples, 2)
    """
    return PCA(Y_2D, num_components=2)

def rotate_maps(Y, inclusions):
    """
    Rotate multiple maps such that the x-axis corresponds to the direction of maximum variance,
    controlled by an inclusion parameter which determines which elements within each map are subject
    to rotation.

    Parameters
    ----------
    Y : list of ndarray
        List of maps, each map shape (n_samples, n_dims)
    inclusions : list of ndarray
        List of 0/1 vectors, each vector of length `n_samples` indicating whether the corresponding
        element in a map should be included in rotation.

    Returns
    -------
    list of ndarray
        List of rotated maps, each map shape (n_samples, n_dims)

    Raises
    ------
    ValueError
        If the length of any inclusion vector does not match the number of samples in its corresponding map.
    """
    if len(Y) != len(inclusions):
        raise ValueError("The length of the inclusions list must match the number of maps.")
    rotated_maps = []
    for map, inclusion in zip(Y, inclusions):
        if len(map) != len(inclusion):
            raise ValueError("The length of each inclusion vector must match the number of samples in its corresponding map.")
        Y_rotated = np.zeros_like(map)
        included_indices = (inclusion == 1)
        if np.any(included_indices):  # Only perform rotation if there are included elements
            Y_rotated[included_indices, :] = rotate_map(map[included_indices, :])
        rotated_maps.append(Y_rotated)
    return rotated_maps