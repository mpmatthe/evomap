"""
Useful transformations after maps have been generated.
"""
from scipy.linalg import orthogonal_procrustes
import numpy as np

def align_maps(Xs, X_ref):
    """ Align a sequence of maps to a reference map via Orthogonal 
    Procrustes Analysis.

    Parameters
    ----------
    Xs : list of ndarrays, each of shape (n_samples, n_dims)
        List of map coordinates
    X_ref : ndarray, shape (n_samples, n_dims)
        Reference map to which all other maps should be aligned to

    Returns
    -------
    list of ndarrays, each of shape (n_samples, n_dims)
        List of aligned map coordinates
    """
    Xs_aligned = []
    n_periods = len(Xs)
    for t in range(n_periods):
        Xs_aligned.append(align_map(Xs[t], X_ref))
    
    return Xs_aligned

def align_map(X, X_ref):
    """ Align a map to a reference map via Orthogonal Procrustes Analysis.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_dims)
        Map coordinates
    X_ref : ndarray of shape (n_samples, n_dims)
        Reference map
    
    Returns
    -------
    ndarray, of shape (n_samples, n_dims)
        Aligned map
    """

    X_orig = X.copy()
    R, _ = orthogonal_procrustes(X_orig, X_ref)

    X_aligned = np.matmul(X_orig, R)
    return X_aligned