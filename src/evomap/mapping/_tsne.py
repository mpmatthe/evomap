""" 
T-Distributed Stochastic Neighborhood Embedding, as propsoed in:

Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).
"""
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._optim import gradient_descent_with_momentum
import inspect

EPSILON = 1e-12

class TSNE():

    def __init__(
        self,
        n_dims = 2, 
        perplexity = 15,
        n_iter = 2000,  
        stop_lying_iter = 250,
        early_exaggeration = 4,
        initial_momentum = .5, 
        final_momentum = .8,
        eta = 'auto',
        n_iter_check = 50,
        init = None, 
        verbose = 0, 
        input_type = 'distance', 
        max_halves = 5, 
        tol = 1e-3,  
        n_inits = 1, 
        step_size = 1
    ):

        self.n_dims = n_dims
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.stop_lying_iter = stop_lying_iter
        self.early_exaggeration = early_exaggeration
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.eta = eta
        self.n_iter_check = n_iter_check
        self.init = init
        self.verbose = verbose
        self.input_type = input_type
        self.max_halves = max_halves
        self.tol = tol
        self.n_inits = n_inits
        self.step_size = step_size
        self.method_str = "TSNE"

    def __str__(self):
        """Return a string representation of the TSNE instance with key parameters and user-modified values."""
        # Get the signature of the __init__ method
        signature = inspect.signature(self.__init__)
        
        # Get the default values of the parameters from the __init__ method
        defaults = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
        
        # Always show key parameters
        result = f"TSNE(perplexity={self.perplexity}, input_type={self.input_type})"
        
        # Collect all attributes that differ from the default
        changed_attrs = []
        for attr, default_val in defaults.items():
            current_val = getattr(self, attr)
            if current_val != default_val:
                changed_attrs.append(f"{attr}={current_val}")
        
        # If any attributes were changed, append them to the result
        if changed_attrs:
            result += "\nUser-modified attributes: " + ", ".join(changed_attrs)
        
        return result


    def fit(self, X):
        """Fit the TSNE model to the input data, without returning the transformed coordinates.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features) or (n_samples, n_samples)
            The input data. If `input_type` is 'vector', `X` should be the feature 
            vectors of the samples. If `input_type` is 'distance', `X` should be 
            the pairwise distance matrix.

        Returns
        -------
        self : object
            Returns the instance of the TSNE class with the configuration matrix 
            `Y_` stored as an attribute.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit the TSNE model and return the transformed coordinates.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features) or (n_samples, n_samples)
            The input data. If `input_type` is 'vector', `X` should be the feature 
            vectors of the samples. If `input_type` is 'distance', `X` should be 
            the pairwise distance matrix.

        Returns
        -------
        np.array of shape (n_samples, n_dims)
            The transformed coordinates in the reduced-dimensional space.
        
        Raises
        ------
        ValueError
            If the `input_type` is not 'distance' or 'vector'.
        """
        if self.input_type == 'distance':
            D = X
        elif self.input_type == 'vector':
            D = cdist(X,X)
        else:
            raise ValueError("Input type should be 'distance' or 'vector', not {0}".format(self.input_type))

        # Get P-Matrix
        if self.verbose > 1:
            print("[TSNE] Calculating P matrix ...")
        
        P = _check_prepare_tsne(self, X)

        n_samples = P.shape[0]
        n_dims = self.n_dims

        # Initialize
        if self.init is None:
            np.random.seed(0)
            init = np.random.normal(0,1,(n_samples, n_dims))
        else:
            init = self.init

        # Set optimization arguments for early Exaggeration
        opt_args = {
            'init': init,
            'method_str': self.method_str,
            'n_iter': self.stop_lying_iter,
            'n_iter_check': self.n_iter_check,
            'eta': self.eta,
            'momentum': self.initial_momentum,
            'start_iter': 0,
            'min_grad_norm': self.tol,
            'verbose': self.verbose,
            'kwargs': {'P' : P*self.early_exaggeration}
        }

        Y, cost = gradient_descent_with_momentum(_kl_divergence, **opt_args)

        opt_args.update({
            'init': Y,
            'n_iter': self.n_iter,
            'momentum': self.final_momentum,
            'start_iter': self.stop_lying_iter,
            'kwargs': {'P': P}
        })

        Y, cost = gradient_descent_with_momentum(_kl_divergence, **opt_args)

        self.Y_ = Y
        self.cost_ = cost

        return self.Y_
        
def _calc_p_matrix(X, included, input_type, perplexity):
    """Calculate the probability matrix (P-matrix) for t-SNE.

    The P-matrix is a joint probability distribution over pairwise similarities. 
    Depending on the input type, this function calculates the matrix from either 
    feature vectors, distance matrices, or similarity matrices. It also handles 
    cases where certain rows are excluded from the calculation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        The input data, which can either be feature vectors (if `input_type` is 'vector'),
        a distance matrix (if `input_type` is 'distance'), or a similarity matrix 
        (if `input_type` is 'similarity').
    included : ndarray of shape (n_samples,), optional
        A binary array (0/1) indicating whether each sample should be included in the 
        P-matrix calculation. If None, all samples are included by default.
    input_type : str
        Specifies the type of input. Should be one of {'vector', 'distance', 'similarity'}.
    perplexity : float
        The desired perplexity, used for tuning the distribution of the P-matrix. 
        Perplexity determines the effective number of neighbors for each point.

    Returns
    -------
    P : ndarray of shape (n_samples, n_samples)
        The joint probability distribution matrix over pairwise similarities.
    
    Raises
    ------
    AssertionError
        If any rows of the input matrix contain only zeros, which would indicate 
        invalid data for the calculation.
    ValueError
        If the input type is not recognized or if the P-matrix contains invalid values.
    """
    from evomap.mapping.evomap import _utils # Import here to avoid circular dependency when initializing the module
    n = X.shape[0]
    if included is None:
        included = np.ones(n)

    if input_type == 'vector':
        X = X[included == 1, :]
        D = cdist(X, X)
        P = _utils._binary_search_perplexity(D.astype(np.float32), perplexity, 0)
        for i in range(n):
            if included[i] == 0:
                P = np.insert(P, i, 0*np.ones((1,P.shape[1])), 0)
                P = np.insert(P, i, 0*np.ones((1, P.shape[0])), 1)

    elif input_type == 'similarity':
        assert np.all(np.sum(X, axis = 0) != 0), "Zero Row(s) in Input Matrix"
        P = X

    elif input_type == 'distance':
        assert np.all(np.sum(X, axis = 0) != 0), "Zero Row(s) in Input Matrix"
        D = X
        D = D[included == 1, :][:, included == 1]
            
        P = _utils._binary_search_perplexity(D.astype(np.float32), perplexity, 0)

        for i in range(n):
            if included[i] == 0:
                P = np.insert(P, i, 0*np.ones((1,P.shape[1])), 0)
                P = np.insert(P, i, 0*np.ones((1, P.shape[0])), 1)

    P = cond_to_joint(P)
    P = np.maximum(P, EPSILON)
    np.fill_diagonal(P,0)

    assert np.all(np.isfinite(P)), "All probabilities should be finite. \
        Check if object at index {0} has sufficient non-zero \
            neighbors.".format(np.where(np.isfinite(P) == False)[0][0])

    assert np.all(P >= 0), "All probabilities should be non-negative"
    assert np.all(P <= 1), ("All probabilities should be less than or equal \
            to one")        

    return P        

def _check_prepare_tsne(model, X):
    """Check and prepare the input data for t-SNE.

    This function validates and prepares the input data for t-SNE by calculating the 
    appropriate learning rate (`eta`) and generating the P-matrix based on the input data.

    Parameters
    ----------
    model : TSNE or EvoMap
        The t-SNE or EvoMap model instance. The function will check and set the 
        learning rate (`eta`) and other parameters from this model.
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        The input data, which can be feature vectors or a distance matrix.

    Returns
    -------
    P : ndarray of shape (n_samples, n_samples)
        The prepared P-matrix representing pairwise similarities.

    Raises
    ------
    ValueError
        If the learning rate (`eta`) is invalid.
    """
    n_samples = X.shape[0]
    if model.eta == "auto":
        model.eta = n_samples / model.early_exaggeration
        model.eta = np.maximum(model.eta, 50)
    else:
        if not (model.eta > 0):
            raise ValueError("learning_rate 'eta' must be a positive number or 'auto'.")
    P = _calc_p_matrix(X, included = None, input_type = model.input_type, perplexity = model.perplexity)
    return P

@jit(nopython=True)
def sqeuclidean_dist(Y):
    """Calculate the pairwise squared Euclidean distance matrix.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_dims)
        The coordinates of the points in the low-dimensional space.

    Returns
    -------
    D : np.ndarray of shape (n_samples, n_samples)
        The squared Euclidean distance matrix.
    """
    n = Y.shape[0]
    d = Y.shape[1]

    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            for k in range(d):
                D[i,j] += (Y[i,k] - Y[j,k])**2

    return D

@jit(nopython=True)
def calc_q_matrix(Y, inclusions):
    """Calculate the Q-matrix of joint probabilities in low-dimensional space.

    The Q-matrix represents the joint probabilities in the low-dimensional 
    space based on the pairwise Euclidean distances between points. A small 
    constant is added to avoid division by zero. The method also allows 
    excluding certain points from the calculation.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_dims)
        Array of map coordinates in the low-dimensional space.
    inclusions : np.ndarray of shape (n_samples,), optional
        A binary array where 1 indicates the point is included and 0 indicates 
        the point is excluded from the probability matrix. If None, all points 
        are included by default.

    Returns
    -------
    Q : np.ndarray of shape (n_samples, n_samples)
        The joint probability matrix in the low-dimensional space.
    dist : np.ndarray of shape (n_samples, n_samples)
        The squared Euclidean distance matrix used to compute Q.
    """
    n = Y.shape[0]
    if inclusions is None:
        inclusions = np.ones(n)
    dist = sqeuclidean_dist(Y)
    dist += 1
    dist **= -1
    
    # Mask non-included objects
    for idx in np.where(inclusions == 0)[0]:
        dist[idx, :] = 0
        dist[:, idx] = 0
        
    sum_dist = np.sum(dist)

    Q = np.maximum(dist/sum_dist, EPSILON)

    # Also return dist, to avoid recomputing it for the gradient
    return Q, dist

def _kl_divergence(Y, P, compute_error = True, compute_grad = True):
    """Calculate the KL-divergence between high-dimensional and low-dimensional joint probabilities.

    This function computes the Kullback-Leibler (KL) divergence between the 
    joint probability distribution in the high-dimensional space (P) and 
    the low-dimensional space (Q). Optionally, it also computes the gradient 
    of the KL-divergence with respect to the low-dimensional coordinates.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_dims)
        Array of map coordinates in the low-dimensional space.
    P : np.ndarray of shape (n_samples, n_samples)
        The joint probability matrix in the high-dimensional space.
    compute_error : bool, optional
        Whether to compute the KL-divergence value, by default True.
    compute_grad : bool, optional
        Whether to compute the gradient of the KL-divergence, by default True.

    Returns
    -------
    error : float or None
        The KL-divergence value, or None if `compute_error` is False.
    grad : np.ndarray or None
        The gradient of the KL-divergence, or None if `compute_grad` is False.
    """
    Q, dist = calc_q_matrix(Y, None)

    if compute_error:
        n = P.shape[0]
        mask = np.eye(n, dtype = 'bool')
        p = P[~mask]
        q = Q[~mask]
        error = np.sum(p*np.log(p/q))
    else:
        error = None
    
    if compute_grad:
        grad = _kl_divergence_grad(Y, P, Q, dist)
    else:
        grad = None

    return error, grad

@jit(nopython=True)
def _kl_divergence_grad(Y, P, Q, dist):
    """Calculate the gradient of the KL-divergence with respect to the low-dimensional coordinates.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_dims)
        Array of map coordinates in the low-dimensional space.
    P : np.ndarray of shape (n_samples, n_samples)
        The joint probability matrix in the high-dimensional space.
    Q : np.ndarray of shape (n_samples, n_samples)
        The joint probability matrix in the low-dimensional space.
    dist : np.ndarray of shape (n_samples, n_samples)
        The squared Euclidean distance matrix in the low-dimensional space.

    Returns
    -------
    dY : np.ndarray of shape (n_samples, n_dims)
        The gradient of the KL-divergence with respect to the map coordinates.
    """
    # Gradient: dC/dY
    (n_samples, n_dims) = Y.shape
    dY = np.zeros((n_samples, n_dims), dtype = np.float32)
    
    PQd = (P - Q) * dist
    for i in range(n_samples):
        dY[i] = np.dot(np.ravel(PQd[i]), Y[i] - Y)

    dY *= 4.0

    return dY

@jit(nopython=True)
def cond_to_joint(P):
    """Convert a conditional probability matrix to a symmetric joint probability matrix.

    This function takes an asymmetric conditional probability matrix (P) and 
    converts it into a symmetric joint probability matrix by averaging the 
    pairwise probabilities and normalizing the result.

    Parameters
    ----------
    P : np.ndarray of shape (n_samples, n_samples)
        The conditional probability matrix.

    Returns
    -------
    P : np.ndarray of shape (n_samples, n_samples)
        The symmetric joint probability matrix.
    """
    np.fill_diagonal(P,0)
    P = P + P.T
    sum_P = np.maximum(np.sum(P), EPSILON)
    P = np.maximum(P / sum_P, EPSILON)
    return P 