"""
Module for evaluating maps.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist, cosine
from scipy.stats import pearsonr

def misalign_score(X_t, normalize=True):
    """
    Calculate misalignment of a sequence of maps.

    Misalignment is measured as the average Euclidean distance between objects' subsequent map positions.
    The final score is averaged across all objects.

    Parameters
    ----------
    X_t : list of ndarrays, each of shape (n_samples, n_dims)
        Map coordinates.
    normalize : bool, optional
        If True, misalignment is normalized by the average interobject distance on the map.
        Useful for comparing maps across differently scaled coordinate systems, by default True.

    Returns
    -------
    float
        Misalignment score, bounded within [0, inf).
        Lower values indicate better alignment.
    """
    n_periods = len(X_t)
    if n_periods < 2:
        raise ValueError("At least two maps are required to compute misalignment.")

    misalignment = 0
    for t in range(1, n_periods):
        distances = np.linalg.norm(X_t[t] - X_t[t-1], axis=1)
        if normalize:
            normalization_factor = np.mean(pdist(X_t[t-1]))
            distances /= normalization_factor if normalization_factor > 0 else 1

        misalignment += np.mean(distances)

    misalignment /= (n_periods - 1)
    return misalignment

def align_score(X_t):
    """
    Calculate alignment of a sequence of maps.

    Alignment is measured as the mean cosine similarity of objects' subsequent
    map positions. The final score is averaged across all objects. Cosine similarity
    measures the cosine of the angle between two vectors, thus providing a scale-invariant
    metric of similarity.

    Parameters
    ----------
    X_t : list of ndarray, each of shape (n_samples, n_dims)
        Sequence of map coordinates.

    Returns
    -------
    float
        Alignment score, bounded between [-1,1].
        Higher values indicate better alignment.
    """
    if len(X_t) < 2:
        raise ValueError("At least two maps are required to compute alignment.")

    n_periods = len(X_t)
    n_samples = X_t[0].shape[0]
    total_cosine_similarity = 0

    for t in range(1, n_periods):
        X_this = X_t[t]
        X_prev = X_t[t-1]
        period_cosine_similarity = 0

        for i in range(n_samples):
            # Compute the cosine similarity for each pair of vectors
            # Note that 'cosine' function returns the cosine distance, so 1 - distance gives similarity
            period_cosine_similarity += 1 - cosine(X_this[i, :], X_prev[i, :])

        # Average cosine similarity across all samples for the current period
        total_cosine_similarity += period_cosine_similarity / n_samples

    # Average the total cosine similarity across all periods
    mean_alignment = total_cosine_similarity / (n_periods - 1)

    return mean_alignment

def hitrate_score(X, D, n_neighbors=10, inc=None, input_format='dissimilarity'):
    """
    Calculate the Hitrate of nearest neighbor recovery for a single map. The 
    score is averaged across all objects. 

    Parameters
    ----------
    X : ndarray of shape (n_samples, d)
        Map coordinates.
    D : ndarray
        Input data, either a similarity/dissimilarity matrix of shape 
        (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10.
    inc : ndarray of shape (n_samples,), optional
        Inclusion array, indicating if an object is present (via 0 and 1s), by default None.
    input_format : str, optional
        One of 'vector', 'similarity', or 'dissimilarity', indicating the type of the input D, by default 'dissimilarity'.

    Returns
    -------
    float
        Hitrate of nearest neighbor recovery, bounded within [0,1].
        Higher values indicate better recovery.

    Raises
    ------
    ValueError
        If the input dimensions mismatch or unsupported input format is provided.
    """
    D = D.copy()
    if n_neighbors < 1:
        raise ValueError('Number of neighbors must be at least 1.')
    if not input_format in ['similarity', 'dissimilarity', 'vector']:
        raise ValueError('Input format should be "vector", "similarity", or "dissimilarity".')

    n_samples = X.shape[0]
    if inc is not None:
        if len(inc) != n_samples:
            raise ValueError('Inclusion array size must match the number of samples in X.')
        if np.any(~np.isin(inc, [0, 1])):
            raise ValueError('Inclusion array must contain only 0 or 1 values.')
        X = X[inc == 1, :]
        D = D[inc == 1, :][:, inc == 1]
        n_samples = X.shape[0]  # Update n_samples after applying inc

    if input_format == 'vector':
        D = cdist(D, D)  # Compute distance matrix from feature vectors

    # Prepare distance matrix for hitrate calculation
    if input_format in ['dissimilarity', 'vector']:
        np.fill_diagonal(D, np.inf)  # Ensure no point is its own neighbor
    else:  # 'similarity'
        np.fill_diagonal(D, -np.inf)  # Ensure no point is its own neighbor

    # Compute distances in the map space
    Dist_map = squareform(pdist(X, 'sqeuclidean'))
    np.fill_diagonal(Dist_map, np.inf)  # Similarly, ensure no point is its own neighbor

    hit_rate = 0
    for i in range(n_samples):
        # Find indices of the n_neighbors closest neighbors in both original and map spaces
        if input_format in ['dissimilarity', 'vector']:
            nearest_original = np.argsort(D[i, :])[:n_neighbors]
        else:  # 'similarity'
            nearest_original = np.argsort(-D[i, :])[:n_neighbors]
        
        nearest_map = np.argsort(Dist_map[i, :])[:n_neighbors]

        # Calculate the intersection of the two neighbor lists
        hit_rate += len(np.intersect1d(nearest_original, nearest_map))

    return hit_rate / (n_neighbors * n_samples)

def adjusted_hitrate_score(X, D, n_neighbors=10, inc=None, input_format='dissimilarity'):
    """
    Calculate the Hitrate of nearest neighbor recovery for a single map, adjusted
    for random agreement. The score is averaged across all objects.

    Parameters
    ----------
    X : ndarray
        Map coordinates, shape (n_samples, d).
    D : ndarray
        Input data, either a similarity/dissimilarity matrix of shape
        (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10.
    inc : ndarray of shape (n_samples,), optional
        Inclusion array, indicating if an object is present (via 0 and 1s), by default None.
    input_format : str, optional
        One of 'vector', 'similarity', or 'dissimilarity', by default 'dissimilarity'.

    Returns
    -------
    float
        Adjusted Hitrate of nearest neighbor recovery, bounded within [0,1].
        Higher values indicate better recovery. Adjusted hitrate corrects the raw
        hitrate by the expected hitrate due to chance.

    Raises
    ------
    ValueError
        If parameters are out of expected range or input dimensions mismatch.
    """
    # First, calculate the raw hitrate score using the previously defined function
    raw_hitrate = hitrate_score(X, D, n_neighbors, inc, input_format)
    
    if inc is not None:
        # Update n_samples to reflect the number of included samples
        n_samples = len(X[inc == 1, :])
    else:
        n_samples = X.shape[0]

    # Calculate the expected hitrate due to chance
    expected_hitrate = n_neighbors / (n_samples - 1) # n-1 because the object itself cannot be a neighbor

    # Adjust the hitrate by subtracting the expected hitrate due to chance
    adjusted_hitrate = raw_hitrate - expected_hitrate

    return adjusted_hitrate

def avg_hitrate_score(X_t, D_t, n_neighbors=10, inc_t=None, input_format='dissimilarity'):
    """
    Calculate the average Hitrate of nearest neighbor recovery for a sequence of maps. 
    The score is averaged across all maps within the sequence.

    Parameters
    ----------
    X_t : list of ndarray
        List of map coordinates for each time period, each of shape (n_samples, d).
    D_t : list of ndarray
        List of input data matrices for each time period, each either a similarity/dissimilarity 
        matrix of shape (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate for each map, by default 10.
    inc_t : list of ndarray, optional
        List of inclusion arrays for each time period, each indicating if an object is present (via 0 and 1s), 
        by default None. If provided, each should match the number of samples in the corresponding X and D.
    input_format : str, optional
        Specifies the input format of D_t, one of 'vector', 'similarity', or 'dissimilarity', by default 'dissimilarity'.

    Returns
    -------
    float
        Average hitrate of nearest neighbor recovery, bounded between [0,1]. Higher values indicate better recovery.

    Raises
    ------
    ValueError
        If there are inconsistencies in array sizes or unsupported input format is specified.
    """
    if not X_t or not D_t or len(X_t) != len(D_t):
        raise ValueError("List of map coordinates and data matrices must be non-empty and of equal length.")
    if inc_t and len(inc_t) != len(X_t):
        raise ValueError("Inclusion arrays, if provided, must match the number of map coordinate arrays.")

    total_hitrate = 0
    n_periods = len(X_t)
    
    for t in range(n_periods):
        inc = inc_t[t] if inc_t else None
        hit_rate = hitrate_score(X=X_t[t], D=D_t[t], n_neighbors=n_neighbors, inc=inc, input_format=input_format)
        total_hitrate += hit_rate

    return total_hitrate / n_periods

def avg_adjusted_hitrate_score(X_t, D_t, n_neighbors=10, inc_t=None, input_format='dissimilarity'):
    """
    Calculate the average Adjusted Hitrate of nearest neighbor recovery for a sequence of maps, 
    adjusted for random agreement. The score is averaged across all maps within the sequence.

    Parameters
    ----------
    X_t : list of ndarray
        List of map coordinates for each time period, each of shape (n_samples, d).
    D_t : list of ndarray
        List of input data matrices for each time period, each either a similarity/dissimilarity 
        matrix of shape (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    n_neighbors : int, optional
        Number of neighbors considered when calculating the adjusted hitrate for each map, by default 10.
    inc_t : list of ndarray, optional
        List of inclusion arrays for each time period, each indicating if an object is present (via 0 and 1s), 
        by default None. If provided, each should match the number of samples in the corresponding X and D.
    input_format : str, optional
        Specifies the input format of D_t, one of 'vector', 'similarity', or 'dissimilarity', by default 'dissimilarity'.

    Returns
    -------
    float
        Average adjusted hitrate of nearest neighbor recovery, bounded between [0,1]. Higher values indicate better recovery.

    Raises
    ------
    ValueError
        If there are inconsistencies in array sizes or unsupported input format is specified.
    """
    if not X_t or not D_t or len(X_t) != len(D_t):
        raise ValueError("List of map coordinates and data matrices must be non-empty and of equal length.")
    if inc_t and len(inc_t) != len(X_t):
        raise ValueError("Inclusion arrays, if provided, must match the number of map coordinate arrays.")

    total_adjusted_hitrate = 0
    n_periods = len(X_t)
    
    for t in range(n_periods):
        inc = inc_t[t] if inc_t else None
        adjusted_hit_rate = adjusted_hitrate_score(X=X_t[t], D=D_t[t], n_neighbors=n_neighbors, inc=inc, input_format=input_format)
        total_adjusted_hitrate += adjusted_hit_rate

    return total_adjusted_hitrate / n_periods

def persistence_score(X_t):
    """
    Calculate persistence of a sequence of maps as the average Pearson correlation coefficient 
    between objects' subsequent map movements (i.e., the first differences of their map positions). 
    The score is averaged across all objects.

    Parameters
    ----------
    X_t : list of ndarrays, each of shape (n_samples, n_dims)
        Sequence of map coordinates. Each ndarray represents a map at a different time.

    Returns
    -------
    float
        Persistence score, bounded within (-1,1).
        Higher positive values indicate higher persistence of map movements across time periods.

    Raises
    ------
    ValueError
        If fewer than two maps are provided or if maps do not have consistent dimensions.
    """
    if len(X_t) < 2:
        raise ValueError("Persistence can only be computed for a sequence of at least two maps.")
    
    # Calculate first differences between consecutive maps
    deltas = np.diff(X_t, axis=0)

    # Calculate correlation of these differences from one time step to the next
    n_periods = len(X_t) - 1  # Number of periods with valid differences
    correlations = []

    for d in range(deltas.shape[2]):  # Iterate over each dimension
        # Flatten the differences in this dimension for all samples across all periods
        flat_deltas = deltas[:, :, d].flatten()
        if np.all(flat_deltas == 0):
            # If there is no change in this dimension, treat the persistence as perfect
            correlations.append(1.0)
        else:
            # Calculate the Pearson correlation between consecutive periods
            corr, _ = pearsonr(flat_deltas[:-X_t[0].shape[0]], flat_deltas[X_t[0].shape[0]:])
            correlations.append(corr)

    # Average the correlations across all dimensions to get the overall persistence score
    return np.mean(correlations)