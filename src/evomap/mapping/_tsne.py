""" 
T-Distributed Stochastic Neighborhood Embedding, as propsoed in:

Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).
"""

import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._optim import gradient_descent_with_momentum

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

    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        
        if self.input_type == 'distance':
            # Note: When using sammon with pre-computed distances, note that
            # optimzation can sometimes be tricky (gradient norm increases 
            # inverse proportionally to distance size. Thus, very small distances
            # can let the gradient explode). 
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
        best_cost = cost

        return self.Y_
        
def _calc_p_matrix(X, included, input_type, perplexity):

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
    """ Check and, if necessary, prepare data for t-SNE.

    Parameters
    ----------
    model: TSNE or EvoMap
        Model object.

    Returns
    ------
    ndarray of shape (n_samples, n_samples)
        Prepared input data
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
    """Calculate Q-Matrix of joint probabilities in low-dim space.
    
    Arguments:
        Y {np.ndarray} -- (n,2) array of map coordinates
        exclusions {np.ndarray} -- condensed-dist-mat indices for exclusions

    Returns:
        Q {np.ndarray} -- (n,n) array of joint probabilities in low-dim space.
        dist {np.ndarray} -- (n,n) array of squared euclidean distances
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
    """ Calculate gradient of KL-divergence dC/dY.
    
    Arguments:
        Y {np.ndarray} -- (n,2) array of map coordinates
        P {np.ndarray} -- condensed-matrix of joint probabilities (high dim)
        Q {np.ndarray} -- condensed-matrix of joint probabilities (low dim)
        dist {np.ndarray} -- condensed-matrix of sq.euclidean distances
    
    Returns:
        dY {np.ndarray} -- (n,2) array of gradient values
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
    """ Take an asymmetric conditional probability matrix and convert it to a 
    symmetric joint probability matrix.

    Symmetrizes and normalizes the matrix. 
    """ 
    np.fill_diagonal(P,0)                  # Set diagonal to zero
    P = P + P.T
    sum_P = np.maximum(np.sum(P), EPSILON)
    P = np.maximum(P / sum_P, EPSILON)
    return P 