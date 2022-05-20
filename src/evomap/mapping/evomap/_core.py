from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import copy
from itertools import product 
from numba import jit

class EvoMap():
    """ EvoMap Interface. Implements default functions shared by all  
    implementation in its child classes. 
    """
    def __init__(self, alpha = 0, p = 1, weighting = 'exponential'):
        self.alpha = alpha
        self.p = p
        self.weighting = weighting
        self.method_str = "" # Overriden by child class

    def get_params(self):
        """Get model parameters."""
        return self.__dict__.items()

    def set_params(self, params):
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def _calc_weights(self, Xs):
        """Calculate object-specific weights, applied to the temporal penalties
        in EvoMap's cost function.

        Parameters
        ----------
        Xs : list of ndarrays, each of shape (n_samples, n_samples) 
            Input data (typically, distances matrices).

        Returns
        -------
        ndarray of shape (n_samples * n_periods, 1)
            Object specific weights, stacked on top of each other for all periods

        """
        n_samples = Xs[0].shape[0]
        n_periods = len(Xs)
        W = np.zeros((n_samples))
        if self.weighting is None:
            W += 1
        else:

            for t in range(1, n_periods):
                delta = Xs[t] - Xs[t-1]
                delta = np.power(delta, 2)
                delta = delta.sum(axis = 1)
                delta = delta.reshape(n_samples)
                W += delta
                
            if self.weighting == 'inverse':
                W = np.power(W, -1)
            elif self.weighting == 'inverse_plus':
                W = np.power(W+1, -1)
            elif self.weighting == 'mirror':
                W = np.max(W) - W
            elif self.weighting == 'exponential':
                if np.max(W) > 1e-12:
                    lamb = 1/np.max(W) # If all goes wrong, readjust to lamb = 5 / np.max(W)
                    W = np.exp(-(lamb * W))
                else:
                    W = np.ones_like(W)
            else:
                raise ValueError("Unkown weighting scheme: {}.".format(self.weighting))

        # Normalize to [0,1]:
        if np.max(W) > 1e-12:
            W = W / np.max(W)
        else:
            print("Weights not calculated. Input data might be fully static.")
            W = np.ones_like(W)
            
        W_all_periods = np.tile(W, n_periods).reshape((n_periods*n_samples, 1))
        return W_all_periods

    def _initialize(self, Xs):
        """Create initialized positions for EvoMap.

        Parameters
        ----------
        Xs : list of ndarrays
            Input data

        Returns
        -------
        ndarray of shape(n_samples * n_periods, n_dims)
            Initialized starting positions.
        """

        n_samples = Xs[0].shape[0]
        n_periods = len(Xs)
        if self.init is None:
            init = np.random.normal(0,.1,(n_samples, self.n_dims))
            init = np.concatenate([init]*n_periods, axis = 0)
        else:
            # Stack list of initialization arrays
            init = np.concatenate(self.init, axis = 0)

        return init


    def _validate_input(self, Xs):
        """ Validate input data vis-a-vis model parameters.

        Parameters
        ----------
        Xs : list of ndarrays
            Input data.

        """
        if not type(Xs) is list:
            raise ValueError('Invalid input data format! Should be a list of equally sized data matrices.')
        n_periods = len(Xs)
        for t in range(1,n_periods):
            if Xs[t].shape != Xs[t-1].shape:
                raise ValueError('Unequal shaped input data!') 
        n_samples = Xs[0].shape[0]
        
        if not self.init is None:
            if not type(self.init) is list:
                raise ValueError('Invalid input type for init! Should be list.')
            for t in range(n_periods):
                if self.init[t].shape[0] != n_samples or self.init[t].shape[1] != self.n_dims:
                    raise ValueError('Invalid shape for init at time {0}! Should be shaped (n_samples, n_dims), but has shape {1}'.format(t, self.init[t].shape)) 

    @staticmethod
    def _calc_static_cost(
        Xs, Y_all_periods, static_cost_function, args = None, kwargs = None):
        """ Calculate total static cost, i.e. sum over all static cost values
        across all periods.

        Parameters
        ----------
        Xs : list of ndarrays, containing the input data
            Sequence of input data (typically, distance matrices)
        Y_all_periods : ndarray of shape (n_samples * n_periods, n_dims)
            Map positions estimated by EvoMap
        static_cost_function : callable
            Static cost function
        args : list, optional
            Additional arguments passed to the static cost function, by default None
        kwargs : dict, optional
            Additional keyword arguments passed to the static cost function, by default None

        Returns
        -------
        float
            Total static cost
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        cost = 0
        n_periods = len(Xs)
        n_samples = Xs[0].shape[0]
        kwargs.update({'compute_grad': False, 'compute_error': True})
        for t in range(n_periods):
            Y_t = _get_positions_for_period(Y_all_periods, n_samples, t)
            cost_t, _ = static_cost_function(Y_t, Xs[t], *args, **kwargs)
            cost += cost_t
        return cost

    def grid_search(
        self, Xs, param_grid, eval_functions = None, eval_labels = None, 
        kwargs = None):
        """Fit model once for each parameter combination within the grid and 
        evaluate each run accordings to various metrics.

        Parameters
        ----------
        Xs : list of ndarrays
            Sequence of input data (typically, distance matrices)
        param_grid : dict
            Parameter grid
        eval_functions : list of callables, optional
            Evaluation functions. Should take a sequence of map layouts 'Ys' 
            as first argument, by default None
        eval_labels : list of strings, optional
            Labels for each evaluation function, by default None
        kwargs : dict, optional
            Additional keyword arguments passed to the evaluation functions, by default None

        Returns
        -------
        DataFrame
            Results for each parameter combination

        """
        self._validate_input(Xs)

        if kwargs is None:
            kwargs = {}

        kwargs['Xs'] = Xs
        if eval_labels is None:
            eval_labels = ['Metric_' + str(i+1) for i in range(len(eval_functions))]

        model = self
        if model.verbose > 0:
            print("[{0}] Evaluating parameter grid..".format(model.method_str))

        def iter_grid(p):
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

        if not 'p' in param_grid.keys():
            param_grid.update({'p': [1]})

        df_res = pd.DataFrame()
        for param_combi in iter_grid(param_grid):
            if model.verbose > 0:
                print("[{0}] .. evaluating parameter combination: ".format(model.method_str) + str(param_combi))
            model_i = copy.deepcopy(model)
            model_i.set_params(param_combi)
            model_i.set_params({'verbose': 0})
            Ys_i = model_i.fit_transform(Xs)
            df_i = {
                'alpha': param_combi['alpha'], 
                'p': int(param_combi['p']), 
                'cost_static_avg': model_i.cost_static_avg_}

            if not eval_functions is None:
                for i, eval_fun in enumerate(eval_functions):
                    kwargs_i = {}
                    for key in kwargs.keys():
                        if key in eval_fun.__code__.co_varnames:
                            kwargs_i[key] = kwargs[key]

                    df_i[eval_labels[i]] = eval_fun(Ys_i, **kwargs_i)

            df_i = pd.DataFrame(df_i, index = [0])
            df_res = pd.concat([df_res, df_i],ignore_index = True, axis = 0)

        df_res.reset_index()
        if model.verbose > 0:
                print("[{0}] Grid Search Completed.".format(model.method_str))

        return df_res

    def fit(self, Xs):
        # Placeholder, needs to overridden by child class.
        raise NotImplementedError

    def fit_transform(self, Xs):
        # Placeholder, needs to overriden by child class.
        raise NotImplementedError

def _evomap_cost_function(
    Y_all_periods, 
    Ds, 
    static_cost_function, 
    weights, 
    alpha, 
    p,
    compute_error = True, 
    compute_grad = True, 
    args = None, 
    kwargs = None):
    """ EvoMap's cost function for a given static cost function.

    Parameters
    ----------
    Y_all_periods : ndarray of shape (n_samples * n_periods, n_dims)
        Map coordinates for all periods, stacked on top of each other.
    Ds : list of ndarrays, each of shape (n_samples,n_samples)
        Sequence of input data (typically, distances matrices).
    static_cost_function : callable
        Static cost function. Needs to return the cost function value and the 
        gradient. Can take additional args / kwargs. The first two arguments 
        should be the map coordinates Y and the input data D.  
    weights : ndarray of shape (n_samples * n_periods, 1)
        Object specific weights.
    alpha : float
        Hyperparamter alpha, controlling the strength of alignment. 
    p : int
        Hyperparameter p, controlling the degree of smoothing.
    compute_error : bool, optional
        True if cost function value should be computed, by default True
    compute_grad : bool, optional
        True if gradient should be computed, by default True
    args : list, optional
        Additional arguments passed to the static cost function, by default None
    kwargs : dict, optional
        Additional keyword arguments passed to the static cost function, by default None

    Returns
    -------
    float
        Cost function value

    ndarray of shape (n_samples * n_periods, n_dims)
        Gradient
    """

    if args is None:
        args = []

    if kwargs is None:
        kwargs = {}

    n_periods = len(Ds)
    n_samples = Ds[0].shape[0]
    n_dims = Y_all_periods.shape[1]

    for t in range(n_periods):
        Y_t = Y_all_periods[(t*n_samples):((t+1)*n_samples), :]

    if compute_error:
        
        # Add all static cost values
        cost = 0
        kwargs.update({'compute_grad': False, 'compute_error': True})
        for t in range(n_periods):
            Y_t = _get_positions_for_period(Y_all_periods, n_samples, t)
            cost_t, _ = static_cost_function(Y_t, Ds[t], *args, **kwargs)
            cost += cost_t

        # Calculate temporal cost
        temp_cost, _ = _evomap_temporal_cost_function(Y_all_periods, n_periods, weights, alpha, p, **kwargs)
        cost += temp_cost
    else:
        cost = None
    
    if compute_grad:
        grad = np.zeros_like(Y_all_periods)
        # Compute all static gradient components
        kwargs.update({'compute_grad': True, 'compute_error': False})
        for t in range(n_periods):
            Y_t = _get_positions_for_period(Y_all_periods, n_samples, t)
            _, grad_t = static_cost_function(Y_t, Ds[t], *args, **kwargs)
            # Stack gradients for each period below each other
            grad[(t*n_samples):((t+1)*n_samples), :] = grad_t

        _, temp_grad = _evomap_temporal_cost_function(Y_all_periods, n_periods, weights, alpha, p, **kwargs)
        grad += temp_grad

    else:
        grad = None
    
    return cost, grad

@jit(nopython=True)
def _get_positions_for_period(Y_all_periods, n_samples, period):
    """Extract map coordinates for period t from the array of all coordinates.  

    Parameters
    ----------
    Y_all_periods : ndarray of shape (n_samples*n_periods, n_dims)
        All map coordinates, stacked on top of each other. 
    n_samples : int
        Number of objects
    period : int
        Period for which coordinates should be extracted.

    Returns
    -------
    ndarray of shape (n_samples, n_dims)
        Map coordinates at focal period
    """
    return Y_all_periods[(period*n_samples):((period+1)*n_samples), :]

@jit(nopython = True)
def _calc_kth_order_dist(Y_all_periods, n_periods, p):
    """Calculate all kth-order distances (up to order p).

    Parameters
    ----------
    Y_all_periods : ndarray of shape (n_samples*n_periods, n_dims)
        All map corrdinates, stacked on top of each other.
    n_periods : int
        Number of periods
    p : int
        Highest order of distances that should be computed.

    Returns
    -------
    list
        List of ndarrays, each of shape (n_samples * n_periods, n_dims), 
        containing the kth order distances at index k. 
    """
    kth_order_dists = []
    n_samples = int(Y_all_periods.shape[0] / n_periods)

    for k in range(p+1):
        if k == 0:
            kth_dist = Y_all_periods
        else:
            kth_dist = np.zeros_like(Y_all_periods)
            for t in range(k, n_periods):
                kdist_prev = _get_positions_for_period(kth_order_dists[k-1], n_samples, t-1)
                kdist_this = _get_positions_for_period(kth_order_dists[k-1], n_samples, t)
                kth_dist[(t*n_samples):((t+1)*n_samples),:] = kdist_this - kdist_prev 
        kth_order_dists.append(kth_dist)

    return kth_order_dists

@jit(nopython=True)
def _shift_elements(vector):
    """Shift all elements in a vector by one (first element becomes zero). 

    Parameters
    ----------
    vector : ndarray
        Input vector

    Returns
    -------
    ndarray
        Vector with all elements shifted by one index (row)
    """
    last_element = 0
    new_vector = np.zeros_like(vector)
    for i in range(len(vector)):
        new_vector[i] = last_element
        last_element = vector[i]
    return new_vector

@jit(nopython=True)
def _calc_kth_order_dist_grad(n_periods, p):
    """Calculate the gradients of all kth-order distances, up to order p.

    Parameters
    ----------
    n_periods : int
        Number of periods.
    p : int
        Highest order of distances.

    Returns
    -------
    list of ndarrays, each of shape (n_periods, n_dims)
        Gradients of kth-order distances at index k.
    """
    partial_delta_k = []
    for k in range(p+1):
        # Partial of delta k (for each k) w.r.t. t (for each t)
        partial_delta_k.append(np.zeros(n_periods))

    for k in range(1,p+1):
        if k == 1:
            partial_delta_k[k][0] = 1 # partial of 1st Order Distance w.r.t (t)
            partial_delta_k[k][1] = -1 # partial of 1st Order distance w.r.t. (t+1)
        else:
            partial_delta_k[k] = partial_delta_k[k-1] - _shift_elements(partial_delta_k[k-1])

    return partial_delta_k

def _evomap_temporal_cost_function(
    Y_all_periods, n_periods, weights, alpha, p, compute_error = True, 
    compute_grad = True):
    """ Calculate temporal component of EvoMap's cost function. 

    Parameters
    ----------
    Y_all_periods : ndarray of shape (n_samples*n_periods, n_dims)
        Map coordinates for all periods, stacked on top of each other.
    n_periods : int
        Number of periods.
    weights : ndarray of shape (n_samples, n_periods, 1)
        Object-specific weights.
    alpha : float
        Hyperparameter alpha, controlling the strength of alignment.
    p : int
        Hyperparameter p, controlling the degree of smoothing.
    compute_error : bool, optional
        True, if cost function value should be computed, by default True
    compute_grad : bool, optional
        Ture, if gradient should be computed, by default True

    Returns
    -------
    float
        Cost function value

    ndarray of shape (n_samples * n_periods, n_dims)
        Gradient
    """
    
    n_samples = int(Y_all_periods.shape[0] / n_periods)
    n_dims = Y_all_periods.shape[1]
    kth_order_dists = _calc_kth_order_dist(Y_all_periods, n_periods, p)
    if compute_error:
        error = 0
        
        for k in range(1, p+1):
            error += alpha * np.sum(np.linalg.norm(weights * kth_order_dists[k], axis = 1)**2)
            # Veriied with numerical example from paper!
            
    else:
        error = None
    
    if compute_grad:
        grad = np.zeros_like(Y_all_periods)
        partial_delta_k = _calc_kth_order_dist_grad(n_periods, p)
        for t in range(n_periods):
            dyn_grad = np.zeros((n_samples, n_dims))
            for k in range(1,p+1):  
                # Skip k = 0, since all partials are zero
                for tau in range(p+1):
                    if (t+tau) < n_periods:
                        dyn_grad += 2 * partial_delta_k[k][tau] * _get_positions_for_period(kth_order_dists[k], n_samples, t+tau)

            grad[(t*n_samples):((t+1)*n_samples), :] = dyn_grad

        grad = alpha * weights * grad 

    else:
        grad = None
    
    return error, grad