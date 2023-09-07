"""
Useful functions to evaluate maps.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist, cosine
from scipy.stats import pearsonr

def misalign_score(X_t, normalize = True):
    """ Calculate misalignment of a sequence of maps.
    
    Misaligned is measured as the average Euclidean distance
    between objects' subsequent map positions. The final score is averaged across
    all objects.

    Parameters
    ----------
    Ys : list of ndarrays, each of shape (n_samples, d)
        Map coordinates.
    normalize : bool, optional
        If true, misalignment is normalized by the average interobject distance 
        on the map. Useful for comparing maps across differently scaled coordinate
         systems, by default True.

    Returns
    -------
    float
        Misalignment score, bounded within [0, inf). 
        Lower values indicate better alignment.
    """
    n_samples = X_t[0].shape[0]
    misalignment = np.zeros(n_samples)
    n_periods = len(X_t)
    for t in range(1, n_periods):
        X_this = X_t[t]
        X_prev = X_t[t-1]
        D = cdist(X_this, X_prev)
        misalignment_t = np.diag(D)
        if normalize:
            misalignment_t = misalignment_t / np.mean(cdist(X_prev, X_prev))

        misalignment += misalignment_t

    misalignment /= (n_periods-1)
    return misalignment.mean()

def align_score(X_t):
    """ Calculate alignment of a sequence of maps.
    
    Alignment is measured as the mean cosine similarity of objects' subsequent 
    map positions. The final score is averaged across all objects. 

    Parameters
    ----------
    Ys : list of ndarrays, each of shape (n_samples, d)
        Map coordinates.

    Returns
    -------
    float
        Alignment score, bounded between [-1,1]. 
        Higher values indicate better alignment.
    """

    mean_alignment = 0
    n_samples = X_t[0].shape[0]
    n_periods = len(X_t)
    for t in range(1, n_periods):
        X_this = X_t[t]
        X_prev = X_t[t-1]
        for i in range(n_samples):
            mean_alignment += 1 - cosine(X_this[i, :], X_prev[i, :])
    
    mean_alignment /= n_samples
    mean_alignment /= (n_periods-1)

    return mean_alignment

def hitrate_score(X, D, n_neighbors = 10, inc = None, input_format = 'dissimilarity'):
    """ Calculate Hitrate of nearest neighbor recovery for a single map. The 
    score is averaged across all objects. 

    Parameters
    ----------
    D : ndarray
        Input data, either a similarity / dissimilarity matrix of shape 
        (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    X : ndarray of shape (n_samples, d)
        Map coordinates.
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    inc : ndarray of shape (n_samples,), optional
        Inclusion array, indicating if an object is present (via 0 and 1s), by default None
    input_format : str, optional
        One of 'vector', 'similarity', or 'dissimilarity', by default 'dissimilarity'

    Returns
    -------
    float
        Hitrate of nearest neighbor recovery, bounded within [0,1]. 
        Higher values indicate better recovery.
    """
    
    hit_rate = 0
    n_samples = X.shape[0]
    if not X.shape[0] == D.shape[0]:
        raise ValueError('Inconsistent array sizes.')
    if not input_format in ['similarity', 'dissimilarity', 'vector']:
        raise ValueError('Input type should be similarity, dissimilarity or vector.')

    # Need to copy the matrix, else "np.fill_diagonal" will modify the original 
    # one
    D = D.copy() 
    if input_format == 'vector':
        # Turn X into a distance matrix
        D = cdist(D,D)

    if not inc is None:
        if np.any(~np.logical_or(inc == 0, inc == 1)):
            raise ValueError('Inclusions should only be 0 or 1.')
        if len(inc) != n_samples:
            raise ValueError('Incosistent array sizes.')
        X = X[inc==1, :]
        D = D[inc==1, :][:, inc == 1]

    Dist_map = squareform(pdist(X, "sqeuclidean"))

    # Make diagonal (self-dissimilarity) larger than any other dissimilarity 
    # (thereby, an object never appears as its own nearest neighbor)
    np.fill_diagonal(Dist_map, np.max(Dist_map)+1e8)
    
    if input_format == 'dissimilarity' or input_format == 'vector':
        # X is a dissimilarity matrix
        # Make diagonal (self-dissimilarity) larger than any other dissimilarity (thereby, an object never appears as its own nearest neighbor)
        np.fill_diagonal(D, np.max(D)+1e8)

        for i in range(n_samples):
            # Sort i-th row of dissimilarity matrix (low-to-high)
            nn_data = np.argsort(D[i,:])[:n_neighbors] 
            nn_map = np.argsort(Dist_map[i, :])[:n_neighbors]

            nn_intersec = [id for id in nn_data if id in nn_map]
            hit_rate +=  len(nn_intersec)

        hit_rate = hit_rate / (n_neighbors * n_samples)

    elif input_format == 'similarity':
        # X is a similarity matrix
        # For similarities, make diagonal (= self similarities) smaller than any other similarity (see above)
        np.fill_diagonal(D, 0)
        for i in range(n_samples):
            # Find max similarity (rather than min dissimilarity)
            nn_data = np.argsort(D[i,:])[-n_neighbors:]
            nn_map = np.argsort(Dist_map[i, :])[:n_neighbors]

            nn_intersec = [id for id in nn_data if id in nn_map]
            hit_rate +=  len(nn_intersec)

        hit_rate = hit_rate / (n_neighbors * n_samples)

    return hit_rate

def adjusted_hitrate_score(X, D, n_neighbors = 10, inc = None, input_format = 'dissimilarity'):
    """ Calculate Hitrate of nearest neighbor recovery for a single map, adjusted
    for random agreement. The score is averaged across all objects.
    
    Parameters
    ----------
    X : ndarray
        Input data, either a similarity / dissimilarity matrix of shape 
        (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    Y : ndarray of shape (n_samples, d)
        Map coordinates.
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    inc : ndarray of shape (n_samples,), optional
        Inclusion array, indicating if an object is present (via 0 and 1s), by default None
    input_format : str, optional
        One of 'vector', 'similarity', or 'dissimilarity', by default 'dissimilarity'

    Returns
    -------
    float
        Adjusted Hitrate of nearest neighbor recovery, bounded within [0,1]. 
        Higher values indicate better recovery.
    """
    
    hitrate = hitrate_score(X = X, D = D, n_neighbors= n_neighbors, inc = inc, input_format=input_format)
    n_samples = X.shape[0]
    adj_hitrate = hitrate - n_neighbors / (n_samples -1)
    return adj_hitrate


def avg_hitrate_score(X_t, D_t, n_neighbors = 10, inc_t = None, input_format = 'dissimilarity'):
    """ Calculate average Hitrate of nearest neighbor recovery for a sequence of 
    maps. The score is averaged across all maps within the sequence. 

    Parameters
    ----------
    Xs : list of ndarrays
        Input data, either in the form of dissimilarity/similarity matrices, each of 
        shape (n_samples, n_samples), or or feature vectors of shape (n_samples, d_input). 
    Ys : list of ndarays, each of shape (n_samples, d)
        _description_
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    Inc_ts : list of ndarays, each of shape (n_samples,), optional
        List of inclusion arrays, indicating if an object is present in a 
        given period (via 0 and 1s), by default None
    input_format : str, optional
        One of 'vector', 'similarity', or 'distance', by default 'dissimilarity'

    Returns
    -------
    float
        Average hitrate, bounded between [0,1]. Higher values indicate better recovery.
    """
    avg_hitrate = 0
    n_periods = len(X_t)
    for t in range(n_periods):
        if inc_t is None:
            inc = None
        else:
            inc = inc_t[t]

        avg_hitrate += hitrate_score(
            X = X_t[t], 
            D = D_t[t], 
            n_neighbors = n_neighbors, 
            inc = inc, 
            input_format = input_format)

    avg_hitrate /= n_periods
    return avg_hitrate

def avg_adjusted_hitrate_score(X_t, D_t, n_neighbors = 10, inc_t = None, input_format = 'dissimilarity'):
    """ Calculate average Hitrate of nearest neighbor recovery for a sequence of 
    maps, adjusted for random agreement. The score is averaged across all 
    maps within the sequence. 

    Parameters
    ----------
    Xs : list of ndarrays
        Input data, either in the form of dissimilarity/similarity matrices, each of 
        shape (n_samples, n_samples), or or feature vectors of shape (n_samples, d_input). 
    Ys : list of ndarays, each of shape (n_samples, d)
        _description_
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    Inc_ts : list of ndarays, each of shape (n_samples,), optional
        List of inclusion arrays, indicating if an object is present in a 
        given period (via 0 and 1s), by default None
    input_format : str, optional
        One of 'vector', 'similarity', or 'dissimilarity', by default 'dissimilarity'

    Returns
    -------
    float
        Average adjusted hitrate, bounded between [0,1]. Higher values 
        indicate better recovery.
    """
    avg_adj_hitrate = 0
    n_periods = len(X_t)
    for t in range(n_periods):
        if inc_t is None:
            inc = None
        else:
            inc = inc_t[t]

        avg_adj_hitrate += adjusted_hitrate_score(
            X = X_t[t], 
            D = D_t[t], 
            n_neighbors = n_neighbors, 
            inc = inc_t, 
            input_format = input_format)

    avg_adj_hitrate /= n_periods
    return avg_adj_hitrate

def persistence_score(X_t):
    """ Calculate persistence of a sequence of maps as the average Pearson 
    correlation coefficient between objects' subsequent map movements (i.e., the
    first differences of their map positions). The score is averaged across all
    objects. 

    Parameters
    ----------
    Ys : list of ndarrays, each of shape (n_samples, 2)
        Map coordinates. 

    Returns
    -------
    float
        Persistence score, bounded within (-1,1). 
        Higher positive values indicate higher persistence.

    """
    
    if len(X_t) < 3:
        raise ValueError("Persistence can only be computed for a sequence of at least three maps.")

    else:
        # Define labels for easier data manipulation 
        labels = [str(i) for i in range(len(X_t[0]))]

        df_delta = pd.DataFrame(columns = ['label', 'x', 'y','t'])
        
        def calc_diffs(df):
            """ Caluclate first differences in map positions. Leaves NAs 
            in the first row for each label (at time t = 0).
            """
            df[['x_diff', 'y_diff']] = df[['x', 'y']].diff()
            df[['x_diff_prev', 'y_diff_prev']] = df[['x_diff', 'y_diff']].shift()
            return df

        for t in range(len(X_t)):
            if X_t[t].shape[1] != 2:
                raise ValueError("Persistence metric is only implemented for 2D map coordinates. Will be extended in a future version.")

            df_delta_t = pd.DataFrame()
            df_delta_t['label'] = labels
            df_delta_t['x'] = X_t[t][:,0]
            df_delta_t['y'] = X_t[t][:,1]
            df_delta_t['t'] = t
            df_delta = pd.concat([df_delta, df_delta_t], axis = 0, sort = True)

        df_delta.index = range(len(df_delta))
        df_delta = df_delta.groupby('label').apply(calc_diffs)
        # Drop NAs in first period where no differences can be calculated
        df_delta = df_delta.dropna(axis = 0, subset = ['x_diff_prev', 'y_diff_prev'])

        if np.sum((df_delta['x_diff_prev'] - df_delta['x_diff'])**2) <= 1e-12:
            print("Warning: Map positions completly static, thus Persistence cannot be calculated!")
            return np.nan
        else:
            x_corr = pearsonr(df_delta['x_diff_prev'] , df_delta['x_diff'])[0]
            y_corr = pearsonr(df_delta['y_diff_prev'] , df_delta['y_diff'])[0]
        return ((x_corr+y_corr)/2)