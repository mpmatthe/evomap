"""
Core functions shared by all implementations.
"""

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
        for t in range(1, n_periods):
            delta = Xs[t] - Xs[t-1]
            delta = np.power(delta, 2)
            delta = delta.sum(axis = 1)
            delta = delta.reshape(n_samples)
            W += delta
            
        if np.max(W) > 1e-12:
            lamb = 1/np.max(W) 
            W = np.exp(-(lamb * W))
        else:
            print("Weights not calculated. Input data might be fully static.")
            W = np.ones_like(W)
           
        W = W.reshape((n_samples, 1))
        return W

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
            init_t = np.zeros((0, self.n_dims))
            for t in range(n_periods):
                init = np.random.normal(0,.1,(n_samples, self.n_dims))
                init_t = np.concatenate([init_t, init], axis = 0)

        else:
            # Stack list of initialization arrays
            init_t = np.concatenate(self.init, axis = 0)

        return init_t


    def _validate_input(self, Xs, inclusions = None):
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
            
            if np.any(np.isnan(Xs[0])):
                raise ValueError('Input contains NaN values')

            if np.any(Xs[0]==np.inf):
                raise ValueError('Input contains Inf values')
            
        n_samples = Xs[0].shape[0]
        
        if not inclusions is None:
            if not type(inclusions) is list:
                raise ValueError('Invalid format for inclusions! Should be a list of (n_samples,) shaped arrays.')
            if not len(inclusions) == len(Xs):
                raise ValueError('Unequal number of inclusion and input data arrays!')
            for t, inc_array in enumerate(inclusions):
                if np.any(~np.isin(inc_array,[0,1])):
                    raise ValueError('Inclusion arrays should only contain 0/1 entries!')
                if len(inc_array) != Xs[t].shape[0]:
                    raise ValueError('Inclusion array at period {0} does not match size of input data!'.format(t))

        if not self.init is None:
            if not type(self.init) is list:
                raise ValueError('Invalid input type for init! Should be list.')
            for t in range(n_periods):
                if self.init[t].shape[0] != n_samples or self.init[t].shape[1] != self.n_dims:
                    raise ValueError('Invalid shape for init at time {0}! Should be shaped (n_samples, n_dims), but has shape {1}'.format(t, self.init[t].shape)) 

    @staticmethod
    def _calc_static_cost(
        Xs, Y_all_periods, static_cost_function, inclusions = None, args = None, kwargs = None, static_cost_kwargs = None):
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
        inclusions: list of np.arrays of shape (n_samples)
            Sequence of 0/1 arrays indicating if an object is included in 
            the estimation
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
        cost_ts = []
        for t in range(n_periods):
            Y_t = _get_positions_for_period(Y_all_periods, n_samples, t)
            if not inclusions is None:
                Y_t = Y_t[inclusions[t] == 1, :]
                X_t = Xs[t][inclusions[t] == 1, :][:, inclusions[t] == 1]
            else:
                X_t = Xs[t]
            if static_cost_kwargs is None:
                static_cost_kwargs = {}
            cost_t, _ = static_cost_function(Y_t, X_t, *args, **kwargs, **static_cost_kwargs)
            cost += cost_t
            cost_ts.append(cost_t)
        return cost, cost_ts

    def grid_search(
        self, Xs, param_grid, inclusions = None, eval_functions = None, 
        eval_labels = None, kwargs = None):
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
        self._validate_input(Xs, inclusions)

        if kwargs is None:
            kwargs = {}

        kwargs['D_t'] = Xs
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
            Ys_i = model_i.fit_transform(Xs, inclusions)
            df_i = {
                'alpha': param_combi['alpha'], 
                'p': int(param_combi['p']), 
                'cost_static_avg': model_i.cost_static_avg_}
            params = [item[0] for item in param_grid.items()]
            params.remove('alpha')
            params.remove('p')
            
            for param in params:
                df_i.update({param: param_combi[param]})

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
        df_res.set_index('alpha', inplace = True)

        if model.verbose > 0:
                print("[{0}] Grid Search Completed.".format(model.method_str))

        return df_res

    def fit(self, Xs, inclusions = None):
        # Placeholder, needs to overridden by child class.
        raise NotImplementedError

    def fit_transform(self, Xs, inclusions = None):
        # Placeholder, needs to overriden by child class.
        raise NotImplementedError
        
def _evomap_cost_function(
    Y_all_periods, 
    Ds, 
    static_cost_function, 
    weights, 
    alpha, 
    p,
    inclusions = None,
    compute_error = True, 
    compute_grad = True,
    static_cost_kwargs = None, 
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
    inclusions: list of n_period np.arrays of shape (n_samples)
        Sequence of inclusion arrays, each containing (n_samples) 0/1 entries
        indicating if an object should be included in the estimation
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


#    for t in range(n_periods):
#        Y_t = Y_all_periods[(t*n_samples):((t+1)*n_samples), :]

    if compute_error:
        
        # Add all static cost values
        cost = 0
        kwargs.update({'compute_grad': False, 'compute_error': True})
        for t in range(n_periods):
            Y_t = _get_positions_for_period(Y_all_periods, n_samples, t)
            
            # If necessary, drop excluded observations before calculating the cost function
            if not inclusions is None:
                Y_t = Y_t[inclusions[t] == 1, :]
                D_t = Ds[t][inclusions[t] == 1, :][:, inclusions[t]==1]
            else:
                D_t = Ds[t]

            if static_cost_kwargs is None:
                static_cost_kwargs = {}
            cost_t, _ = static_cost_function(Y_t, D_t, *args, **kwargs, **static_cost_kwargs)
            cost += cost_t

        # Calculate temporal cost
        temp_cost, _ = _evomap_temporal_cost_function(
            Y_all_periods, n_periods, weights, alpha, p, inclusions, **kwargs)
        cost += temp_cost

    else:
        cost = None
    
    if compute_grad:
        grad = np.zeros_like(Y_all_periods)
        # Compute all static gradient components
        kwargs.update({'compute_grad': True, 'compute_error': False})
        for t in range(n_periods):
            Y_t = _get_positions_for_period(Y_all_periods, n_samples, t)

            # If necessary, drop exluded observations
            full_grad_t = np.zeros_like(Y_t)
            if not inclusions is None:
                Y_t = Y_t[inclusions[t]==1,:]
                D_t = Ds[t][inclusions[t] == 1,:][:,inclusions[t] == 1]
                _, grad_t = static_cost_function(Y_t, D_t, *args, **kwargs, **static_cost_kwargs)
                full_grad_t[inclusions[t]==1, :] = grad_t
            else:
                D_t = Ds[t]  

                _, full_grad_t = static_cost_function(Y_t, D_t, *args, **kwargs, **static_cost_kwargs)




            # Stack gradients for each period below each other
            grad[(t*n_samples):((t+1)*n_samples), :] = full_grad_t

        _, temp_grad = _evomap_temporal_cost_function(Y_all_periods, n_periods, weights, alpha, p, inclusions, **kwargs)
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
    Y_all_periods, n_periods, weights, alpha, p, inclusions = None, 
    compute_error = True, compute_grad = True):
    """ Calculate temporal component of EvoMap's cost function. 

    Parameters
    ----------
    Y_all_periods : ndarray of shape (n_samples*n_periods, n_dims)
        Map coordinates for all periods, stacked on top of each other.
    n_periods : int
        Number of periods.
    weights : ndarray of shape (n_samples, 1)
        Object-specific weights.
    alpha : float
        Hyperparameter alpha, controlling the strength of alignment.
    p : int
        Hyperparameter p, controlling the degree of smoothing.
    inclusions: list of n_periods np.arrays of shape (n_samples)
        Sequence of arrays with 0/1 entries indicating if an object should 
        be included in the estimation
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
            for t in range(k, n_periods):
                dist_kt = _get_positions_for_period(kth_order_dists[k], n_samples, t)
                if not inclusions is None:
                    inc_tk = inclusions[t].copy()
                    # check if object was present in time t and k preceding periods
                    for z in range(t,t-k-1,-1):
                        inc_tk *= inclusions[z]
                    dist_kt[inc_tk == 0,:] = 0
                dist_kt = dist_kt * weights

                error += alpha * np.sum(np.linalg.norm(dist_kt, axis = 1)**2)
            
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
                        if not inclusions is None:
                            dyn_grad[inclusions[t+tau] == 0,:] = 0
            grad[(t*n_samples):((t+1)*n_samples), :] = dyn_grad * weights

        grad = alpha * grad 

    else:
        grad = None
    
    return error, grad