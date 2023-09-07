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


def PCA(X , num_components):
     
    #Step-1
    X_meaned = X - np.mean(X , axis = 0)
     
    #Step-2
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    #Step-3
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    #Step-5
    eigenvector_subset = sorted_eigenvectors[:,0:num_components]
     
    #Step-6
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced

def rotate_map(Y_2D):
    Y_PCA = PCA(Y_2D, num_components=2)
    return Y_PCA

def rotate_maps(Y, inclusions):
    """Rotate a map (or sequence of maps) such that the x-axis corresponds to 
    the direction of maximum variance.

    Parameters
    ----------
    Y : _type_
        _description_
    """

    n_samples = Y[0].shape[0]
    n_dims = Y[0].shape[1]
    n_periods = len(Y)
    Y = np.vstack(Y)
    Y_rotated = np.zeros_like(Y)
    inc = np.hstack(inclusions)
    Y_rotated[inc == 1, :] = rotate_map(Y[inc==1, :])
    Y_rotated_list = []
    for t in range(n_periods):
        Y_rotated_list.append(Y_rotated[t*n_samples:(t+1)*n_samples, :])

    return Y_rotated_list

