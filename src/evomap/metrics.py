"""
Useful functions to evaluate market maps.

Author: Maximilian Matthe <matthe@wiwi.uni-frankfurt.de>
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist, cdist, cosine
from scipy.stats import pearsonr

def misalign_score(Ys, normalize = True):
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
    n_samples = Ys[0].shape[0]
    misalignment = np.zeros(n_samples)
    n_periods = len(Ys)
    for t in range(1, n_periods):
        Y_this = Ys[t]
        Y_prev = Ys[t-1]
        D = cdist(Y_this, Y_prev)
        misalignment_t = np.diag(D)
        if normalize:
            misalignment_t = misalignment_t / np.mean(cdist(Y_prev, Y_prev))

        misalignment += misalignment_t

    misalignment /= (n_periods-1)
    return misalignment.mean()

def align_score(Ys):
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
    n_samples = Ys[0].shape[0]
    n_periods = len(Ys)
    for t in range(1, n_periods):
        Y_this = Ys[t]
        Y_prev = Ys[t-1]
        for i in range(n_samples):
            mean_alignment += 1 - cosine(Y_this[i, :], Y_prev[i, :])
    
    mean_alignment /= n_samples
    mean_alignment /= (n_periods-1)

    return mean_alignment

def hitrate_score(X, Y, n_neighbors = 10, inc = None, input_type = 'similarity'):
    """ Calculate Hitrate of nearest neighbor recovery for a single map. The 
    score is averaged across all objects. 

    Parameters
    ----------
    X : ndarray
        Input data, either a similarity / distance matrix of shape 
        (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    Y : ndarray of shape (n_samples, d)
        Map coordinates.
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    inc : ndarray of shape (n_samples,), optional
        Inclusion array, indicating if an object is present (via 0 and 1s), by default None
    input_type : str, optional
        One of 'vector', 'similarity', or 'distance', by default 'similarity'

    Returns
    -------
    float
        Hitrate of nearest neighbor recovery, bounded within [0,1]. 
        Higher values indicate better recovery.
    """
    
    hit_rate = 0
    n_samples = X.shape[0]
    if not Y.shape[0] == X.shape[0]:
        raise ValueError('Inconsistent array sizes.')
    if not input_type in ['similarity', 'distance', 'vector']:
        raise ValueError('Input type should be similarity, distance or vector.')

    # Need to copy the matrix, else "np.fill_diagonal" will modify the original 
    # one
    X = X.copy() 
    if input_type == 'vector':
        # Turn X into a distance matrix
        X = cdist(X,X)

    if not inc is None:
        if np.any(~np.logical_or(inc == 0, inc == 1)):
            raise ValueError('Inclusions should only be 0 or 1.')
        if len(inc) != n_samples:
            raise ValueError('Incosistent array sizes.')
        Y = Y[inc==1, :]
        X = X[inc==1, :][:, inc == 1]

    Dist_map = squareform(pdist(Y, "sqeuclidean"))

    # Make diagonal (self-distances) larger than any other distance 
    # (thereby, an object never appears as its own nearest neighbor)
    np.fill_diagonal(Dist_map, np.max(Dist_map)+1e8)
    
    if input_type == 'distance' or input_type == 'vector':
        # X is a distance matrix
        # Make diagonal (self-distances) larger than any other distance (thereby, an object never appears as its own nearest neighbor)
        np.fill_diagonal(X, np.max(X)+1e8)

        for i in range(n_samples):
            # Sort i-th row of distance matrix (low-to-high)
            nn_data = np.argsort(X[i,:])[:n_neighbors] 
            nn_map = np.argsort(Dist_map[i, :])[:n_neighbors]

            nn_intersec = [id for id in nn_data if id in nn_map]
            hit_rate +=  len(nn_intersec)

        hit_rate = hit_rate / (n_neighbors * n_samples)

    elif input_type == 'similarity':
        # X is a similarity matrix
        # For similarities, make diagonal (= self similarities) smaller than any other similarity (see above)
        np.fill_diagonal(X, 0)
        for i in range(n_samples):
            # Find max similarity (rather than min distance)
            nn_data = np.argsort(X[i,:])[-n_neighbors:]
            nn_map = np.argsort(Dist_map[i, :])[:n_neighbors]

            nn_intersec = [id for id in nn_data if id in nn_map]
            hit_rate +=  len(nn_intersec)

        hit_rate = hit_rate / (n_neighbors * n_samples)

    return hit_rate

def adjusted_hitrate_score(X, Y, n_neighbors = 10, inc = None, input_type = 'similarity'):
    """ Calculate Hitrate of nearest neighbor recovery for a single map, adjusted
    for random agreement. The score is averaged across all objects.
    
    Parameters
    ----------
    X : ndarray
        Input data, either a similarity / distance matrix of shape 
        (n_samples, n_samples), or a matrix of feature vectors of shape (n_samples, d_input).
    Y : ndarray of shape (n_samples, d)
        Map coordinates.
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    inc : ndarray of shape (n_samples,), optional
        Inclusion array, indicating if an object is present (via 0 and 1s), by default None
    input_type : str, optional
        One of 'vector', 'similarity', or 'distance', by default 'similarity'

    Returns
    -------
    float
        Adjusted Hitrate of nearest neighbor recovery, bounded within [0,1]. 
        Higher values indicate better recovery.
    """
    
    hitrate = hitrate_score(X = X, Y = Y, n_neighbors= n_neighbors, inc = inc, input_type=input_type)
    n_samples = X.shape[0]
    adj_hitrate = hitrate - n_neighbors / (n_samples -1)
    return adj_hitrate


def avg_hitrate_score(Ys, Xs, n_neighbors = 10, Inc_ts = None, input_type = 'distance'):
    """ Calculate average Hitrate of nearest neighbor recovery for a sequence of 
    maps. The score is averaged across all maps within the sequence. 

    Parameters
    ----------
    Xs : list of ndarrays
        Input data, either in the form of distance/similarity matrices, each of 
        shape (n_samples, n_samples), or or feature vectors of shape (n_samples, d_input). 
    Ys : list of ndarays, each of shape (n_samples, d)
        _description_
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    Inc_ts : list of ndarays, each of shape (n_samples,), optional
        List of inclusion arrays, indicating if an object is present in a 
        given period (via 0 and 1s), by default None
    input_type : str, optional
        One of 'vector', 'similarity', or 'distance', by default 'similarity'

    Returns
    -------
    float
        Average hitrate, bounded between [0,1]. Higher values indicate better recovery.
    """
    avg_hitrate = 0
    n_periods = len(Xs)
    for t in range(n_periods):
        if Inc_ts is None:
            inc_t = None
        else:
            inc_t = Inc_ts[t]

        avg_hitrate += hitrate_score(
            X = Xs[t], 
            Y = Ys[t], 
            n_neighbors = n_neighbors, 
            inc = inc_t, 
            input_type = input_type)

    avg_hitrate /= n_periods
    return avg_hitrate

def avg_adjusted_hitrate_score(Ys, Xs, n_neighbors = 10, Inc_ts = None, input_type = 'distance'):
    """ Calculate average Hitrate of nearest neighbor recovery for a sequence of 
    maps, adjusted for random agreement. The score is averaged across all 
    maps within the sequence. 

    Parameters
    ----------
    Xs : list of ndarrays
        Input data, either in the form of distance/similarity matrices, each of 
        shape (n_samples, n_samples), or or feature vectors of shape (n_samples, d_input). 
    Ys : list of ndarays, each of shape (n_samples, d)
        _description_
    n_neighbors : int, optional
        Number of neighbors considered when calculating the hitrate, by default 10
    Inc_ts : list of ndarays, each of shape (n_samples,), optional
        List of inclusion arrays, indicating if an object is present in a 
        given period (via 0 and 1s), by default None
    input_type : str, optional
        One of 'vector', 'similarity', or 'distance', by default 'similarity'

    Returns
    -------
    float
        Average adjusted hitrate, bounded between [0,1]. Higher values 
        indicate better recovery.
    """
    avg_adj_hitrate = 0
    n_periods = len(Xs)
    for t in range(n_periods):
        if Inc_ts is None:
            inc_t = None
        else:
            inc_t = Inc_ts[t]

        avg_adj_hitrate += adjusted_hitrate_score(
            X = Xs[t], 
            Y = Ys[t], 
            n_neighbors = n_neighbors, 
            inc = inc_t, 
            input_type = input_type)

    avg_adj_hitrate /= n_periods
    return avg_adj_hitrate

def persistence_score(Ys):
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
    
    if len(Ys) < 3:
        raise ValueError("Persistence can only be computed for a sequence of at least three maps.")

    else:
        # Define labels for easier data manipulation 
        labels = [str(i) for i in range(len(Ys[0]))]

        df_delta = pd.DataFrame(columns = ['label', 'x', 'y','t'])
        
        def calc_diffs(df):
            """ Caluclate first differences in map positions. Leaves NAs 
            in the first row for each label (at time t = 0).
            """
            df[['x_diff', 'y_diff']] = df[['x', 'y']].diff()
            df[['x_diff_prev', 'y_diff_prev']] = df[['x_diff', 'y_diff']].shift()
            return df

        for t in range(len(Ys)):
            if Ys[t].shape[1] != 2:
                raise ValueError("Persistence metric is only implemented for 2D map coordinates. Will be extended in a future version.")

            df_delta_t = pd.DataFrame()
            df_delta_t['label'] = labels
            df_delta_t['x'] = Ys[t][:,0]
            df_delta_t['y'] = Ys[t][:,1]
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