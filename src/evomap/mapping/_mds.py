"""
Stress-Based Multidimensional Scaling.
"""
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._core import gradient_descent_line_search

EPSILON = 1e-12

class MDS():

    def __init__(
        self,
        n_dims = 2, 
        metric = True,  
        n_iter = 2000,  
        n_iter_check = 50,
        init = None, 
        verbose = 0, 
        optim = 'GD', 
        input_type = 'distance', 
        maxhalves = 5, 
        tol = 1e-3,  
        n_inits = 1, 
        step_size = 1
    ):

        self.n_dims = n_dims
        self.metric = metric
        self.n_iter = n_iter
        self.n_iter_check = n_iter_check
        self.init = init
        self.verbose = verbose
        self.optim = optim
        self.input_type = input_type
        self.maxhalves = maxhalves
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
                print("[{0}] Initialization {1}/{2}".format(self.method_str,i+1, self.n_inits))

            if self.init is None:
                init = np.random.normal(0,.1,(n_samples, n_dims))
            else:
                init = self.init

            if self.optim == 'GD':
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

                Y, cost = gradient_descent_line_search(_normalized_stress_function, **opt_args)
            else:
                raise ValueError("Optimization routine should be 'GD'. SMACOF option not yet implemented.")

            if cost < best_cost:
                self.Y_ = Y
                self.cost_ = cost
                best_cost = cost

        return self.Y_

def _normalized_stress_function(positions, disparities, compute_error = True, compute_grad = True):
    distances = cdist(positions, positions, metric = 'euclidean')
    if compute_error:
        dis = np.triu(distances)
        disp = np.triu(disparities)
        stress = np.sqrt(np.sum((disp - dis)**2)/np.sum(dis**2)) 

    else:
        stress = None

    if compute_grad:
        grad = _normalized_stress_gradient(positions, distances, disparities)
    else: 
        grad = None
    return stress, grad

@jit(nopython=True)
def _normalized_stress_gradient(positions, distances, disparities):
    #TODO: Vectorize calculations and remove jit decorator. Then move both function as an @staticmethod within the class and call them via self.
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

