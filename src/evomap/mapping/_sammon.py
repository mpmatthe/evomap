"""
Nonlinear Sammon Mapping, as proposed in: 

Sammon, J. W. (1969). A nonlinear mapping for data structure analysis. IEEE Transactions on computers, 100(5), 401-409.
"""
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._optim import gradient_descent_line_search
import inspect

class Sammon():

    def __init__(
        self,
        n_dims = 2, 
        n_iter = 2000,  
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
        self.n_iter = n_iter
        self.n_iter_check = n_iter_check
        self.init = init
        self.verbose = verbose
        self.input_type = input_type
        self.max_halves = max_halves
        self.tol = tol
        self.n_inits = n_inits
        self.step_size = step_size
        self.method_str = "SAMMON"

    def __str__(self):
        """Return a string representation of the Sammon instance with input_type and user-modified parameters."""
        # Get the signature of the __init__ method
        signature = inspect.signature(self.__init__)
        
        # Get the default values of the parameters from the __init__ method
        defaults = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
        
        modified_params = []
        
        # Always show the 'input_type' parameter
        result = f"Sammon(input_type={self.input_type}"
        
        # Check if any current attributes differ from their default values and append them
        for param, default_value in defaults.items():
            current_value = getattr(self, param)
            if current_value != default_value and param != "input_type":
                modified_params.append(f"{param}={current_value}")
        
        # If there are modified parameters, append them to the result
        if modified_params:
            result += ", " + ", ".join(modified_params)
        
        result += ")"
        return result

    def fit(self, X):
        """Fit the Sammon model to the input data.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features) or (n_samples, n_samples)
            The input data. If `input_type` is 'vector', `X` should be the feature 
            vectors of the samples. If `input_type` is 'distance', `X` should be 
            the pairwise distance matrix.

        Returns
        -------
        self : object
            Returns the instance of the Sammon class with the configuration matrix 
            `Y_` stored as an attribute.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit the Sammon mapping model and return the transformed coordinates.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features) or (n_samples, n_samples)
            The input data. If `input_type` is 'vector', `X` should be the feature 
            vectors of the samples. If `input_type` is 'distance', `X` should be 
            the pairwise distance matrix.

        Returns
        -------
        np.array of shape (n_samples, n_dims)
            The transformed coordinates of the samples in the reduced-dimensional space.
        
        Raises
        ------
        ValueError
            If the `input_type` is not 'distance' or 'vector'.
        """
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
                'max_halves': self.max_halves,
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
    """Check and prepare the input distance matrix for Sammon Mapping.

    This function ensures that the input distance matrix is valid for Sammon Mapping.
    It checks for strictly positive off-diagonal dissimilarities and adds a small 
    diagonal correction if necessary.

    Parameters
    ----------
    D : ndarray of shape (n_samples, n_samples)
        Input distance matrix. Should be a square, symmetric matrix representing 
        pairwise dissimilarities between samples.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Prepared distance matrix with a diagonal correction applied.

    Raises
    ------
    ValueError
        If any off-diagonal entries in the distance matrix are non-positive.
    """
    n_samples = D[0].shape[0]
    D = D + np.eye(n_samples)     
    if np.count_nonzero(D<=0) > 0:
        raise ValueError("Off-diagonal dissimilarities must be strictly positive")  
    return D

def _sammon_stress_function(positions, disparities, compute_error = True, compute_grad = True):
    """Compute the Sammon stress function and its gradient.

    The Sammon stress function measures the discrepancy between the input distances 
    (disparities) and the distances among the estimated positions in the reduced space. 
    Optionally, it can also compute the gradient of the stress function for optimization purposes.

    Parameters
    ----------
    positions : ndarray of shape (n_samples, n_dims)
        The estimated positions of the samples in the reduced-dimensional space.
    disparities : ndarray of shape (n_samples, n_samples)
        The input distance (or dissimilarity) matrix.
    compute_error : bool, optional
        Whether to compute the stress (error) value, by default True.
    compute_grad : bool, optional
        Whether to compute the gradient of the stress function, by default True.

    Returns
    -------
    float or None
        The Sammon stress value (cost), or None if `compute_error` is False.
    ndarray or None
        The gradient of the stress function, or None if `compute_grad` is False.
    """
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
    """Compute the gradient of the Sammon stress function.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_dims)
        The current positions of the samples in the reduced-dimensional space.
    D_map : ndarray of shape (n_samples, n_samples)
        The pairwise Euclidean distances between the estimated positions.
    D : ndarray of shape (n_samples, n_samples)
        The input distance (or dissimilarity) matrix.

    Returns
    -------
    ndarray of shape (n_samples * n_dims)
        The computed gradient of the Sammon stress function, flattened for optimization.
    """    
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