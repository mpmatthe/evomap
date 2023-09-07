"""
Useful transformation for data pre-processing.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

def diss2sim(diss_mat, transformation = 'inverse', eps = 1e-3):
    """ Transform a dissimilarity matrix to a similarity matrix

    Parameters
    ----------
    diss_mat : ndarray of shape (n_samples, n_samples)
        Matrix of pairwise dissimilarities.
    transformation : str, optional
        Transformation function, either 'inverse' or 'mirror', by default 'inverse'
    eps : float, optional
        Incremental constant to avoid division by zero, by default 1e-3

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Matrix of pairwise similarities.
    """

    if transformation == "inverse":
        sim_mat = 1/(1+diss_mat)

    elif transformation == "mirror":
        # Normalize dissimilarities to [0,1), than mirror it
        diss_mat = diss_mat / (np.max(diss_mat)+eps)
        sim_mat = 1 - diss_mat
        
    else:
        raise ValueError('Unknown transformation type')

    np.fill_diagonal(sim_mat, 1)
    return sim_mat   

def sim2diss(sim_mat, transformation = 'inverse', eps = 1e-4):
    """ Transform a similarity matrix to a dissimilarity matrix.

    Parameters
    ----------
    sim_mat : ndarray of shape (n_samples, n_samples)
        Matrix of pairwise similarities
    transformation : str, optional
        Transformation function, either 'inverse' or 'mirror', by default 'inverse'
    eps : float, optional
        Incremental constant to avoid division by zero, by default 1e-3

    Returns
    -------
    ndaray of shape (n_samples, n_samples)
        Matrix of pairwise dissimilarities.
    """

    if transformation == 'inverse':
        sim_mat = np.maximum(sim_mat, eps)
        diss_mat = (1/sim_mat)

    elif transformation == 'mirror':
        # Normalize similarities to [0,1]       
        if np.max(sim_mat)>1:
            sim_mat = sim_mat / (np.max(sim_mat)+eps)
        
        diss_mat = 1 - sim_mat

    np.fill_diagonal(diss_mat, 0)
    return diss_mat 

def coocc2sim(coocc_mat):
    """ Transform a matrix with co-occurrence counts to a similarity matrix. 

    Parameters
    ----------
    coocc_mat : ndarray of shape (n_samples, n_samples)
        Matrix of co-occurrence counts.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Matrix of pairwise similarities.
    """

    np.fill_diagonal(coocc_mat,0)
    sim_mat = coocc_mat / np.sum(coocc_mat, axis = 1).reshape((-1,1))

    # Make symmetric:
    n = sim_mat.shape[0]
    i_upper = np.triu_indices(n, 1)
    sim_mat[i_upper] = sim_mat.T[i_upper]

    return sim_mat

def edgelist2matrix(df, score_var, id_var_i, id_var_j, time_var = None, time_selected = None):
    """ Transform an edgelist to a relationship matrix.

    Parameters
    ----------
    df : DataFrame
        Data containing the edgelist. Each row should include a pair. Needs to include
        two id variables and a score variable. Can also include a time variable.
    score_var : string
        The score variable. 
    id_var_i : string
        The first id variable.
    id_var_j : string
        The second id variable.
    time_var : string, optional
        The time variable (int), by default None
    time_selected : int, optional
        The selected time, by default None

    Returns
    -------
    S: ndarray of shape (n_samples, n_samples)
        A matrix of pairwise relationships.

    ids: ndarray of shape (n_samles, )
        Identifiers for each element of the matrix.
    """

    if not time_var is None:
        df = df[df[time_var] == time_selected]
    ids = np.unique(np.concatenate([df[id_var_i], df[id_var_j]], axis = 0))
    ids = list(ids)
    n = len(ids)

    df = df[(df[id_var_i].isin(ids)) & (df[id_var_j].isin(ids))]
    row = [ids.index(id) for id in df[id_var_i]]
    col = [ids.index(id) for id in df[id_var_j]]

    scores = list(df[score_var])

    S = coo_matrix((scores, (row, col)), shape=(n, n))
    S = S.toarray()
    S = np.nan_to_num(S, 0)
    return S, np.array(ids)

def edgelist2matrices(df, score_var, id_var_i, id_var_j, time_var):
    """Transform a time-indexed edgelist to a sequence of relationship matrices.

    Parameters
    ----------
    df : DataFrame
        Data containing the edgelist. Each row should include a pair. Needs to include
        two id variables, a score variable, and a time variable. 
    score_var : string
        The score variable. 
    id_var_i : string
        The first id variable.
    id_var_j : string
        The second id variable.
    time_var : string
        The time variable (int)

    Returns
    -------
    S_t: list of ndarrays of shape (n_samples, n_samples) with length (n_periods)
        A sequence of relationship matrices.

    ids_t: ndarray of shape (n_samles, )
        Identifiers for each element of the matrix.
    """
    df = df.sort_values(by = time_var)
    periods = df[time_var].unique()
    S_t = []
    ids_t = []
    for period in periods:
        data_t = df[df[time_var] == period]
        S, ids = edgelist2matrix(data_t, 
                                        score_var = score_var,
                                         id_var_i = id_var_i, 
                                         id_var_j = id_var_j)
        S_t.append(S)
        ids_t.append(ids)

    return S_t, ids_t

def normalize_diss_mat(D):

    max_diss = np.max(D) 
    D_norm = D / max_diss
    return D_norm

def normalize_diss_mats(D_ts):
    """ Normalize a sequence of dissimilarity matrices by a common factor 
    (the max. dissimilarity within the sequence).
    
    Parameters
    ----------
    D_ts : list of ndarrays, each of shape (n_samples, n_samples)
        Sequence of dissimilarity matrices.
    Returns
    -------
    D_ts: ndarray of shape (n_samples, n_samples)
        Sequence of dissimilarity matrices, normalized by the maximum dissimilarity within
        the input sequence.

    """
    n_periods = len(D_ts)
    max_diss = - np.inf
    for t in range(n_periods):
        max_diss_t = np.max(D_ts[t]) 
        if max_diss_t > max_diss:
            max_diss = max_diss_t

    for t in range(n_periods):
        D_ts[t] = D_ts[t] / max_diss
    return D_ts


def expand_matrices(Xts, names_t):
	""" Exand list of similarity matrices to equal shape and calculate inclusion vectors.
	
	Args:

	Returns:
		(list, list, list): list of similarity matrices (equal size), list of inclusion vectors (0/1) and list of all labels.
	"""

	# Step 1: Construct a big similarity matrix
	
	# Get unique list while preserving the order 
	# careful: list(set(list)) does NOT produce a stable ordering (across runs / seeds)!

	all_labels = [[label for label in names] for names in names_t]
	all_labels = [item for sublist in all_labels for item in sublist]
	seen = set()
	seen_add = seen.add
	all_labels = [label for label in all_labels if not (label in seen or seen_add(label))]

	n_periods = len(Xts)
	S_ts = [] # similarity matrices
	Inc_ts = [] # inclusion vectors (initialized with -1)

	for t in range(n_periods):
		Inc_ts.append(np.ones(len(all_labels), dtype = int) * -1)

	for t in range(n_periods):
	# Step 2: Fill it for each period
		S_t = pd.DataFrame(data = np.zeros((len(all_labels), len(all_labels))), index = all_labels, columns = all_labels)
		
		labels_t = names_t[t]

		S_t.loc[labels_t,labels_t] = Xts[t]
		Inc_t = [1*(label in labels_t) for label in all_labels]

		S_ts.append(S_t)
		Inc_ts[t][:] = Inc_t

	S_ts = [S_t.values for S_t in S_ts]

	return S_ts, Inc_ts, all_labels

def calc_distances(X, metric = 'euclidean'):
    """ Caluclate matrix of pairwise distances among the rows of an input 
    matrix. 

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_dims)
        Input matrix.

    metric: string
        The distance metric to use. Can be any of 
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 
        'kulsinski', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 
        'sokalsneath', 'sqeuclidean', 'yule'.

	Returns:
		ndarray of shape (n_samples, n_samples): Matrix of pairwise distances.
	"""

    from scipy.spatial.distance import squareform, pdist
    dist_mat = squareform(pdist(X, metric = metric))
    return dist_mat