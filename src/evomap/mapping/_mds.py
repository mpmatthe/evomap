"""
Stress-Based Multidimensional Scaling.
"""
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._optim import gradient_descent_line_search
from ._cmds import CMDS
from sklearn.linear_model import LinearRegression
from ._regression import IsotonicRegression
import inspect

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

    def __str__(self):
        """Create a string representation of the MDS instance. Displays the key attributes and 
        all parameters modified by the user.

        Returns
        -------
        str
            A summary of the key attributes of this MDS object, including modified parameters.
        """
        # Initial part of the output with always-displayed attributes
        result = f"MDS Type: {self.mds_type}, Input Type: {self.input_type}"

        # Get the signature of the __init__ method
        signature = inspect.signature(self.__init__)
        
        # Get the default values of the parameters from the __init__ method
        defaults = {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}

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
        """Fit the MDS model to the input data, without returning the 
        transformed positions. 

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features) or (n_samples, n_samples)
            The input data. If `input_type` is 'vector', X should be the feature 
            vectors of the samples. If `input_type` is 'distance', X should be 
            a pairwise distance matrix.

        Returns
        -------
        self : object
            The instance of the MDS class, after fitting the model to the input data.
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """Fit the MDS model to the input data and return transformed positions. 

        Dependning on 'input_type', the input data is either interpreted as a distance matrix or feature vectors.
        The method uses gradient descent to optimize the lower-dimensional positions such that a Stress function, 
        measuring the discrepancy between the input distances and resulting configuration, is minimized.

        Parameters
        ----------
        X : np.array of shape (n_samples, n_features) or (n_samples, n_samples)
            The input data. If `input_type` is 'vector', X should be the feature 
            vectors of the samples. If `input_type` is 'distance', X should be 
            a pairwise distance matrix.

        Returns
        -------
        np.array of shape (n_samples, n_dims)
            The transformed positions in the lower-dimensional space.

        Raises
        ------
        ValueError
            If `input_type` is neither 'distance' nor 'vector', a ValueError is raised.
        """
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
                np.random.seed(0)
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
    """Compute normalized stress and its gradient. 

    The stress function quantifies the goodness-of-fit between the input 
    disparities (or distances) and the Euclidean distances in the low-dimensional 
    space, with options for different MDS types: absolute, ratio, interval, and 
    ordinal scaling. The input distances are transformed to disparities according to the mds type. 
    Optionally, the function also computes the gradient to be used in optimization.

    Parameters
    ----------
    positions : np.array of shape (n_samples, n_dims)
        The estimated positions in the low-dimensional space.
    disparities : np.array of shape (n_samples, n_samples)
        The input distances or disparities matrix, depending on the MDS type.
    mds_type : str, optional
        The type of MDS scaling to use: 'absolute', 'ratio', 'interval', 
        or 'ordinal'. If None, 'absolute' scaling is used by default.
    compute_error : bool, optional
        Whether to compute the normalized stress value, by default True.
    compute_grad : bool, optional
        Whether to compute the gradient of the stress function, by default True.

    Returns
    -------
    float or None
        The computed stress value, or None if `compute_error` is False.
    np.array of shape (n_samples, n_dims) or None
        The computed gradient of the stress function, or None if `compute_grad` is False.

    Raises
    ------
    ValueError
        If an invalid `mds_type` is provided, or if `mds_type` is not recognized.
    """
    def normalize_dhat(d_hat, n_samples):
        """Normalize the flattened disparities to match the total variance of the data.

        This method scales the input disparities so that their total sum of 
        squares matches the total number of unique pairwise distances. The 
        normalization ensures that the disparities are comparable to the distances.

        Parameters
        ----------
        d_hat : np.array of shape (n_samples * (n_samples - 1) / 2)
            The flattened array of pairwise disparities.
        n_samples : int
            The number of samples or points in the dataset.

        Returns
        -------
        np.array of shape (n_samples * (n_samples - 1) / 2)
            The normalized disparities.
        """
        return d_hat * np.sqrt((n_samples * (n_samples - 1) / 2) / (d_hat**2).sum())

    def rebuild_matrix(disp_flat, n_samples):
        """Rebuild a full symmetric matrix from a flattened array of lower triangular entries.

        Given a 1D array containing the lower triangular elements of a matrix, 
        this method reconstructs the full symmetric matrix where the upper triangular part mirrors 
        the lower triangular part.

        Parameters
        ----------
        disp_flat : np.array of shape (n_samples * (n_samples - 1) / 2)
            The flattened array of lower triangular matrix entries (disparities).
        n_samples : int
            The number of samples or points, which determines the size of the matrix.

        Returns
        -------
        np.array of shape (n_samples, n_samples)
            The symmetric matrix reconstructed from the lower triangular entries.
        """
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
    """Calculate the gradient of the normalized stress function for MDS.

    Parameters
    ----------
    positions : np.array of shape (n_samples, n_dims)
        The estimated positions in the low-dimensional space.
    distances : np.array of shape (n_samples, n_samples)
        The Euclidean distances among the estimated positions.
    disparities : np.array of shape (n_samples, n_samples)
        The input disparities (or distance) matrix.

    Returns
    -------
    np.array of shape (n_samples, n_dims)
        The gradient of the stress function with respect to the positions.
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