"""
Module for data pre-processing, including transformations between different data formats.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform, pdist

def diss2sim(diss_mat, transformation='inverse', eps=1e-3):
    """
    Transform a dissimilarity matrix to a similarity matrix.

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
    if diss_mat.shape[0] != diss_mat.shape[1]:
        raise ValueError("Dissimilarity matrix must be square.")

    if transformation == "inverse":
        sim_mat = 1 / (1 + diss_mat)
    elif transformation == "mirror":
        max_val = np.max(diss_mat) + eps
        diss_mat_normalized = diss_mat / max_val
        sim_mat = 1 - diss_mat_normalized
    else:
        raise ValueError('Unknown transformation type "{}"'.format(transformation))

    np.fill_diagonal(sim_mat, 1)
    return sim_mat

def sim2diss(sim_mat, transformation='inverse', eps=1e-4):
    """
    Transform a similarity matrix to a dissimilarity matrix.

    Parameters
    ----------
    sim_mat : ndarray of shape (n_samples, n_samples)
        Matrix of pairwise similarities.
    transformation : str, optional
        Transformation function, either 'inverse' or 'mirror', by default 'inverse'.
        'inverse' - Transforms by taking the reciprocal of the similarity scores.
        'mirror' - Transforms by reflecting the similarity scores about 0.5 (1 - similarity).
    eps : float, optional
        Incremental constant to avoid division by zero, by default 1e-4

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Matrix of pairwise dissimilarities.
    """
    if sim_mat.shape[0] != sim_mat.shape[1]:
        raise ValueError("Similarity matrix must be square.")

    # Ensuring the diagonal elements are not below the threshold which could distort the transformation
    np.fill_diagonal(sim_mat, np.maximum(1.0, np.diagonal(sim_mat)))

    if transformation == 'inverse':
        # Ensure no similarity value is less than eps to avoid division by zero
        sim_mat_clipped = np.maximum(sim_mat, eps)
        diss_mat = 1 / sim_mat_clipped
    elif transformation == 'mirror':
        # Normalize similarities if needed
        if np.any(sim_mat > 1):
            sim_mat = sim_mat / (np.max(sim_mat) + eps)
        diss_mat = 1 - sim_mat
    else:
        raise ValueError(f'Unknown transformation type "{transformation}". Valid options are "inverse" or "mirror".')

    np.fill_diagonal(diss_mat, 0)
    return diss_mat

def coocc2sim(coocc_mat):
    """
    Transform a matrix with co-occurrence counts to a similarity matrix.

    Parameters
    ----------
    coocc_mat : ndarray of shape (n_samples, n_samples)
        Matrix of co-occurrence counts. Assumes a non-negative, symmetric matrix where
        the diagonal can be ignored (usually representing self-co-occurrence).

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Matrix of pairwise similarities, normalized such that each element is a proportion
        of the maximum co-occurrence for that row.

    Notes
    -----
    The function sets the diagonal to zero to prevent self-similarity from skewing the results.
    If a row's total co-occurrence is zero, it sets the entire row's similarity to zero to avoid
    division by zero.
    """
    n_samples = coocc_mat.shape[0]
    np.fill_diagonal(coocc_mat, 0)
    row_sums = np.sum(coocc_mat, axis=1)
    row_sums_with_epsilon = np.where(row_sums == 0, np.finfo(float).eps, row_sums)
    sim_mat = coocc_mat / row_sums_with_epsilon[:, np.newaxis]
    sim_mat = (sim_mat + sim_mat.T) / 2

    return sim_mat
def edgelist2matrix(df, score_var, id_var_i, id_var_j, time_var=None, time_selected=None):
    """
    Transform an edgelist to a relationship matrix.
    Parameters
    ----------
    df : DataFrame
        Data containing the edgelist. Each row should include a pair. Must contain
        two id variables and a score variable. Can optionally include a time variable.
    score_var : string
        The score variable.
    id_var_i : string
        The first id variable.
    id_var_j : string
        The second id variable.
    time_var : string, optional
        The time variable, by default None.
    time_selected : int, optional
        The selected time, by default None.
    Returns
    -------
    S: ndarray of shape (n_samples, n_samples)
        A matrix of pairwise relationships.
    ids: ndarray
        Identifiers for each element of the matrix.
    Raises
    ------
    ValueError:
        If required columns are missing in the DataFrame.
    """
    required_columns = {score_var, id_var_i, id_var_j}
    if time_var:
        required_columns.add(time_var)
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Filter data if a specific time is selected
    if time_var and time_selected is not None:
        df = df[df[time_var] == time_selected]
        if df.empty:
            raise ValueError(f"No data found for selected time: {time_selected}")

    # Get unique identifiers and create a mapping index
    unique_ids = np.unique(np.concatenate([df[id_var_i], df[id_var_j]], axis = 0))
    id_index = {id_val: idx for idx, id_val in enumerate(unique_ids)}

    # Prepare the matrix
    n = len(unique_ids)
    row_indices = df[id_var_i].map(id_index).values
    col_indices = df[id_var_j].map(id_index).values
    scores = df[score_var].values

    # Create a sparse matrix and convert to dense
    S = coo_matrix((scores, (row_indices, col_indices)), shape=(n, n), dtype=np.float64).toarray()

    # Symmetrize the matrix
    S = (S + S.T) / 2

    return S, unique_ids

def edgelist2matrices(df, score_var, id_var_i, id_var_j, time_var):
    """
    Transform a time-indexed edgelist into a sequence of relationship matrices.

    Parameters
    ----------
    df : DataFrame
        Data containing the edgelist. Each row should include a pair and must contain
        two id variables, a score variable, and a time variable.
    score_var : string
        The score variable used to assign values in the matrix.
    id_var_i : string
        The first id variable corresponding to rows in the matrix.
    id_var_j : string
        The second id variable corresponding to columns in the matrix.
    time_var : string
        The time variable used to split the data into different matrices.

    Returns
    -------
    S_t : list of ndarray
        A list of relationship matrices, each corresponding to a different time period.
    ids_t : ndarray
        Array of identifiers for each element of the matrices.

    Raises
    ------
    ValueError:
        If the DataFrame is missing any required columns or if there are no valid entries for any time period.
    """
    required_columns = {score_var, id_var_i, id_var_j, time_var}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    # Sort the DataFrame based on the time variable to ensure chronological order
    df_sorted = df.sort_values(by=time_var)

    # Get unique time periods
    periods = df_sorted[time_var].unique()

    # Prepare lists to hold the output matrices and identifiers
    S_t = []
    ids_t = []

    # Process each time period separately
    for period in periods:
        period_data = df_sorted[df_sorted[time_var] == period]

        if period_data.empty:
            raise ValueError(f"No data found for time period: {period}")

        S, ids = edgelist2matrix(period_data, score_var, id_var_i, id_var_j)
        S_t.append(S)
        ids_t.append(ids)

    return S_t, ids_t

def normalize_diss_mat(D):
    """
    Normalize a dissimilarity matrix by the maximum dissimilarity observed in the matrix.
    
    Parameters
    ----------
    D : ndarray of shape (n_samples, n_samples)
        A dissimilarity matrix.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Normalized dissimilarity matrix.

    Raises
    ------
    ValueError
        If the input matrix is not square or if the maximum dissimilarity is zero.
    """
    if D.shape[0] != D.shape[1]:
        raise ValueError("Dissimilarity matrix must be square.")

    max_diss = np.max(D)
    if max_diss == 0:
        raise ValueError("Maximum dissimilarity in the matrix is zero, normalization cannot be performed.")

    # Normalize the matrix by the maximum dissimilarity
    normalized_D = D / max_diss

    return normalized_D

def normalize_diss_mats(D_ts):
    """
    Normalize a sequence of dissimilarity matrices by a common factor 
    (the maximum dissimilarity within the sequence).
    
    Parameters
    ----------
    D_ts : list of ndarray, each of shape (n_samples, n_samples)
        Sequence of dissimilarity matrices.
    
    Returns
    -------
    list of ndarray
        Sequence of dissimilarity matrices, normalized by the maximum dissimilarity within
        the input sequence.

    Raises
    ------
    ValueError
        If any matrix is not square or if the list is empty.
    """
    if not D_ts:
        raise ValueError("Input list of dissimilarity matrices is empty.")

    # Verify that all matrices are square
    if any(mat.shape[0] != mat.shape[1] for mat in D_ts):
        raise ValueError("All dissimilarity matrices must be square.")

    # Find the global maximum dissimilarity across all matrices
    max_diss = max(np.max(mat) for mat in D_ts)
    if max_diss == 0:
        raise ValueError("Maximum dissimilarity across all matrices is zero, cannot normalize.")

    # Normalize all matrices by the global maximum dissimilarity
    normalized_mats = [mat / max_diss for mat in D_ts]

    return normalized_mats


def expand_matrices(X_ts, labels_ts):
    """
    Expand a list of similarity matrices (X_ts) to equal shape and calculate inclusion vectors.

    Args:
        X_ts (list of ndarray): List of similarity matrices for each time point.
        labels_ts (list of list): List of labels corresponding to each matrix in X_ts.

    Returns:
        tuple: Contains a list of expanded similarity matrices, inclusion vectors, and all labels.
    """
    all_labels = [[label for label in names] for names in labels_ts]
    all_labels = [item for sublist in all_labels for item in sublist]
    seen = set()
    seen_add = seen.add
    all_labels = [label for label in all_labels if not (label in seen or seen_add(label))]
    
    expanded_matrices = []
    inclusion_vectors = []

    for X, labels in zip(X_ts, labels_ts):
        full_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)
        matrix_df = pd.DataFrame(X, index=labels, columns=labels)
        full_matrix.update(matrix_df)
        inclusion_vector = np.array([int(label in labels) for label in all_labels])

        expanded_matrices.append(full_matrix.values)
        inclusion_vectors.append(inclusion_vector)

    return expanded_matrices, inclusion_vectors, all_labels

def calc_distances(X, metric='euclidean'):
    """
    Calculate matrix of pairwise distances among the rows of an input matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_dims)
        Input matrix containing samples for which pairwise distances will be calculated.
    metric : str, optional
        The distance metric to use. Can be any of those supported by `scipy.spatial.distance.pdist`,
        such as 'euclidean', 'cityblock', 'cosine', etc. Defaults to 'euclidean'.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        A matrix of pairwise distances, where each element (i, j) is the distance
        between the i-th and j-th rows of the input matrix X according to the specified metric.

    Raises
    ------
    ValueError
        If the metric specified is not supported by `scipy.spatial.distance.pdist`.
    """
    try:
        # Calculate the pairwise distances using pdist and squareform to convert it into a square matrix
        distance_matrix = squareform(pdist(X, metric=metric))
    except ValueError as e:
        raise ValueError(f"The specified metric '{metric}' is not supported. Error: {str(e)}")

    return distance_matrix