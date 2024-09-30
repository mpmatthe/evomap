"""
Stress-Based Multidimensional Scaling.
"""
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._optim import gradient_descent_line_search
from ._cmds import CMDS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from ._regression import IsotonicRegression

EPSILON = 1e-10

class MDS():

    def __init__(
        self,
        n_dims = 2, 
        mds_type = None,  
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
        self.mds_type = mds_type
        self.n_iter = n_iter
        self.n_iter_check = n_iter_check
        self.init = init
        self.verbose = verbose
        self.input_type = input_type
        self.max_halves = max_halves
        self.tol = tol
        self.n_inits = n_inits
        self.step_size = step_size
        self.method_str = "MDS"

    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        
        if self.input_type == 'distance':
            D = X
        elif self.input_type == 'vector':
            D = cdist(X,X)
        else:
            raise ValueError("Input type should be 'distance' or 'vector', not {0}".format(self.input_type))

        n_samples = D.shape[0]
        n_dims = self.n_dims

        # If initialization is provided, use it and run model (once).
        # Else, try n_init different initializiations
        if not self.init is None:
            self.n_inits = 1

        best_cost = np.inf
        for i in range(self.n_inits):
            if self.verbose > 0:
                if self.n_inits > 1:
                    print("[{0}] Initialization {1}/{2}".format(self.method_str,i+1, self.n_inits))

            if self.init is None:
                init = np.random.normal(0,.1,(n_samples, n_dims))
            elif type(self.init) == str and self.init == "cmds":
                init = CMDS().fit_transform(D)
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
                'args': [D],
                'kwargs': {'mds_type': self.mds_type}
            }

            Y, cost = gradient_descent_line_search(_normalized_stress_function, **opt_args)

            if cost < best_cost:
                self.Y_ = Y
                self.cost_ = cost
                best_cost = cost

        # Enforce consistent signs of coordinates
        for dim in range(n_dims):
            # Check the sign of the first element in the dimension
            if self.Y_[0, dim] > 0:
                self.Y_[:, dim] *= -1

        return self.Y_

def _normalized_stress_function(
    positions, disparities, mds_type = None, compute_error = True, 
    compute_grad = True):
    """Compute normalized stress as a measure of goodness-of-fit between 
    input distances and the distances among the estimated positions.

    Parameters
    ----------
    positions : np.array of shape (n_samples, n_dims)
        estimated positions
    disparities : np.array of shape (n_samples, n_samples)
        input distances (or transform disparities)
    inclusions : np.array of shape (n_samples), optional
        array of 0/1 entries indicating if an object should be included
        in the estimation, by default None
    compute_error : bool, optional
        indicates if cost funciton value should be computed, by default True
    compute_grad : bool, optional
        indicates if gradient should be computed, by default True

    Returns
    -------
    float, array of shape (n_samples, n_dims)
        cost function value and gradient
    """

    def normalize_dhat(d_hat, n_samples):
        return d_hat * np.sqrt((n_samples * (n_samples - 1) / 2) / (d_hat**2).sum())

    def rebuild_matrix(disp_flat, n_samples):
        # Take a 1d array of lower triangular entries and rebuild full symmetric matrix
        D_hat = np.zeros((n_samples, n_samples))
        D_hat[np.tril_indices(n_samples,-1)] = disp_flat
        D_hat = D_hat + D_hat.transpose()
        return D_hat

    distances = cdist(positions, positions, metric = 'euclidean')
    n_samples = distances.shape[0]

    if mds_type is None or mds_type == 'absolute':
        disp_hat = disparities

    elif mds_type == 'ratio':
        # Get n * (n-1) / 2 unique distances and disparities
        distances_flat = distances[np.tril_indices(len(distances),-1)]
        disparities_flat = disparities[np.tril_indices(len(disparities),-1)]        

        lr = LinearRegression(positive=True, fit_intercept=False)
        X = disparities_flat.reshape(-1,1)
        y = distances_flat.reshape(-1,1)
        lr = lr.fit(X,y)
        disp_hat = lr.predict(X).reshape(-1)
        disp_hat = normalize_dhat(disp_hat, n_samples)

        # Rebuild matrix of (fitted) disparities
        disp_hat = rebuild_matrix(disp_hat, n_samples)

    elif mds_type == 'interval':
        distances_flat = distances[np.tril_indices(len(distances),-1)]
        disparities_flat = disparities[np.tril_indices(len(disparities),-1)]        

        lr = LinearRegression(positive=True)
        X = disparities_flat.reshape(-1,1)
        y = distances_flat.reshape(-1,1)
        lr = lr.fit(X,y)
        disp_hat = lr.predict(X).reshape(-1)
        disp_hat = normalize_dhat(disp_hat, n_samples)

        # Rebuild matrix of (fitted) disparities
        disp_hat = rebuild_matrix(disp_hat, n_samples)

    elif mds_type == 'ordinal':
        ir = IsotonicRegression()
        distances_flat = distances[np.tril_indices(len(distances),-1)]
        disparities_flat = disparities[np.tril_indices(len(disparities),-1)]        

        disp_hat = ir.fit_transform(X = disparities_flat, y = distances_flat)
        #TOOD: Rename disparities to dissimilarities, before fitting
        disp_hat = disp_hat.reshape(-1)
        disp_hat = normalize_dhat(disp_hat, n_samples)

        # Rebuild matrix of (fitted) disparities
        disp_hat = rebuild_matrix(disp_hat, n_samples)

    else:
        raise ValueError("Unknown MDS type {0}!".format(mds_type))
    
    if compute_grad:
        grad = _normalized_stress_gradient(positions, distances, disp_hat)
    else:
        grad = None
    
    if compute_error:
        dis = np.triu(distances)
        disp = np.triu(disp_hat)
        stress = np.sqrt(np.sum((disp - dis)**2)/np.sum(dis**2)) 

    else:
        stress = None

    return stress, grad

@jit(nopython=True)
def _normalized_stress_gradient(
    positions, distances, disparities):
    """Calculate gradient of normalized stress function.

    Parameters
    ----------
    positions : np.array of shape (n_samples, n_dims)
        estimated postiions
    distances : np.array of shape (n_samples, n_samples)
        euclidean distances among estimated positions
    disparities : np.array of shape (n_samples, n_samples)
        input distance (or disparity) matrix

    Returns
    -------
    np.array of shape (n_samples, n_dims)
        gradient
    """
    n_samples = distances.shape[0]
    n_dims = positions.shape[1]
    gradient = np.zeros(shape = (n_samples, n_dims))
    for i in range(n_samples):
        for l in range(n_dims):
            grad_il = 0
            for j in range(n_samples):
                if j != i:
                    dist_ij = distances[i,j]
                    if dist_ij == 0:
                        dist_ij += EPSILON
                    grad_il -= (1- disparities[i,j] / dist_ij) * (positions[j,l] - positions[i,l])
            gradient[i,l] = grad_il

    return gradient / (n_samples - 1)