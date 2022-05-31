"""
Nonlinear Sammon Mapping, as proposed in: 

Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.
"""

import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._core import gradient_descent_line_search

class Sammon():

    def __init__(
        self,
        n_dims = 2, 
        n_iter = 2000,  
        n_iter_check = 50,
        init = None, 
        verbose = 0, 
        input_type = 'distance', 
        maxhalves = 5, 
        tol = 1e-3,  
        n_inits = 1, 
        step_size = 1
    ):

        self.n_dims = n_dims
        self.n_iter = n_iter
        self.n_iter_check = n_iter_check
        self.init = init
        self.verbose = verbose
        self.input_type = input_type
        self.maxhalves = maxhalves
        self.tol = tol
        self.n_inits = n_inits
        self.step_size = step_size
        self.method_str = "SAMMON"

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

        # Validate input
        D = _check_prepare_input_sammon(D)

        n_samples = D.shape[0]
        n_dims = self.n_dims

        # If initialization is provided, use it and run model (once).
        # Else, try n_init different initializiations
        if not self.init is None:
            self.n_inits = 1

        best_cost = np.inf
        for i in range(self.n_inits):
            if self.verbose > 0:
                print("[{0}] Initialization {1}/{2}".format(self.method_str,i+1, self.n_inits))

            if self.init is None:
                init = np.random.normal(0,1,(n_samples, n_dims))
            else:
                init = self.init

            # Set gradient descent arguments
            opt_args = {
                'init': init,
                'method_str': self.method_str,
                'n_iter': self.n_iter,
                'n_iter_check': self.n_iter_check,
                'step_size': self.step_size,
                'maxhalves': self.maxhalves,
                'min_grad_norm': self.tol,
                'verbose': self.verbose,
                'args': [D]
            }

            Y, cost = gradient_descent_line_search(_sammon_stress_function, **opt_args)

            if cost < best_cost:
                self.Y_ = Y
                self.cost_ = cost
                best_cost = cost

        return self.Y_

def _check_prepare_input_sammon(D):
    """ Check and, if necessary, prepare data for Sammon Mapping.

    Parameters
    ----------
    D : ndarray of shape (n_samples, n_samples)
        Input distance matrix.

    Returns
    ------
    ndarray of shape (n_samples, n_samples)
        Prepared input data
    """
    n_samples = D[0].shape[0]
    D = D + np.eye(n_samples)     
    if np.count_nonzero(D<=0) > 0:
        raise ValueError("Off-diagonal dissimilarities must be strictly positive")  
    return D

def _sammon_stress_function(positions, disparities, compute_error = True, compute_grad = True):
    distances = cdist(positions, positions, metric = 'euclidean')
    if compute_error:    

        n_samples = positions.shape[0]
        D_map = distances.copy()
        D_map = D_map[np.triu_indices(n_samples, k = 1)]
        D_data = disparities[np.triu_indices(n_samples, k = 1)]

        delta = D_data - D_map
        D_data_inv = 1. / D_data
        cost = np.sum((delta**2) * D_data_inv) / np.sum(D_data)
    else:
        cost = None

    if compute_grad:
        grad = _sammon_stress_gradient(positions, distances, disparities)
        # Need to reshape gradient. Raveled for speed optim.
        grad = np.reshape(grad, (-1,positions.shape[1]),order='F')
    else:
        grad = None
        
    return cost, grad

@jit(nopython=True)
def _sammon_stress_gradient(Y, D_map, D):
    
    n_samples = Y.shape[0]
    n_dims = Y.shape[1]
    D_map = D_map + np.eye(n_samples)
    D_map_inv = 1. / D_map
    D_inv = 1. / D

    delta = D_map_inv - D_inv

    one = np.ones((n_samples,n_dims))
    deltaone = np.dot(delta,one) # sum of distances, (reapated n_dims times)

    dY = np.dot(delta,Y) - (Y * deltaone)
    dinv3 = D_map_inv ** 3
    y2 = Y ** 2
    H = np.dot(dinv3,y2) - deltaone - 2*Y * np.dot(dinv3,Y) + y2 * np.dot(dinv3,one)
    iY = dY.transpose().flatten() / np.abs(H.transpose().flatten()) # Gradient / Hessian

    return iY