# EvoMap implemented for Sammon Mapping
#
# References:
#   Sammon, John W. Jr., "A Nonlinear Mapping for Data Structure Analysis", IEEE Transactions on Computers, vol. C-18, no. 5, pp 401-409, May 1969.
#   
# Credits:
#   Optimization code is inspired by the Python implementaion of 
#   (static) Sammon mapping by Tom J. Pollard (https://twitter.com/tompollard)
#   See: https://github.com/tompollard/sammon
# 
# Author: 
#   Maximilian Matthe (matthe@wiwi.uni-frankfurt.de)
#
# This version:
#   November 2021

from numba import jit
import numpy as np 
from scipy.spatial.distance import cdist

from ._core import calc_gradient
from ._core import validate_input
from ._core import calc_dyn_cost
from ._core import init_inclusions
from ._core import build_inclusion_array
from ._core import initialize_positions
from ._core import calc_dyn_gradient
from ._core import extend_gradient
from ._core import calc_weights_new

from itertools import product
import copy

class Error(Exception):
    """Base class for other exceptions"""
    pass

class DivergingGradientError(Error):
    """Raised when the input value is too small"""
    pass


class EvoSammon():
    """EvoSammon implemented for Sammon Mapping. 
    """

    def __init__(
        self,
        alpha = 0,
        p = 1,
        weighted = 'exponential',
        n_dims = 2,
        init = None, 
        verbose = 0,
        maxhalves = 10,
        tol = 1e-5,
        max_iter = 2000,
        input_type = 'distance',
        step_size = .1,
        max_tries = 5):

        self.alpha = alpha
        self.p = p
        self.weighted = weighted
        self.n_dims = n_dims
        if not init is None:
            self.init = init.copy()
        else:
            self.init = init
        self.verbose = verbose
        self.maxhalves = maxhalves
        self.max_iter = max_iter
        self.input_type = input_type
        self.step_size = step_size
        self.tol = tol
        self.max_tries = max_tries


    def get_params(self):
        return self.__dict__.items()

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

    @staticmethod
    @jit(nopython=True)
    def _calc_gradient_sammon(Y, Dinv, D_map_inv):
        
        n_samples = Y.shape[0]
        n_dims = Y.shape[1]

        delta = D_map_inv - Dinv

        one = np.ones((n_samples,n_dims))
        deltaone = np.dot(delta,one) # sum of distances, (reapated n_dims times)

        dY = np.dot(delta,Y) - (Y * deltaone)
        dinv3 = D_map_inv ** 3
        y2 = Y ** 2
        H = np.dot(dinv3,y2) - deltaone - 2 * Y * np.dot(dinv3,Y) + y2 * np.dot(dinv3,one)
        iY = -dY.transpose().flatten() / np.abs(H.transpose().flatten()) # Gradient / Hessian
        # iY: Step (Grad/Hess), dY: Grad
        return iY


    def _find_step_size_via_halving(self, Ys, dYs, Ds, Ws, inclusions):

        C_old,_,_ = self._evaluate_cost_function(Ys, Ds, Ws, inclusions)
        step_size = self.step_size
        for j in range(self.maxhalves):
            Ys_new = Ys + step_size * dYs
            C_new,_,_ = self._evaluate_cost_function(Ys_new, Ds, Ws, inclusions)
            if C_new < C_old:
                break
            else:
                step_size = 0.5 * step_size

        return step_size    

    @staticmethod
    def _get_step_size_sammon(init_step_size, Y_old, dY, D, Dinv, E, maxhalves):
        """ We dont use numba here, because cdist is faster"""
        n_samples = Y_old.shape[0]
        n_dims = Y_old.shape[1]
        step_size = init_step_size
        # Use step-halving procedure to ensure progress is made
        for j in range(maxhalves):
#            dY_reshape = np.reshape(dY, (-1,n_dims),order='F')
            Y = Y_old +  step_size * dY
            D_map = cdist(Y, Y) + np.eye(n_samples)
            delta = D - D_map
            E_new = ((delta**2)*Dinv).sum()
            scale = 0.5 / (D.sum() - np.diag(D).sum())
            E_new = E_new * scale
            if E_new < E:
                break
            else:
                step_size = 0.5*step_size

        return step_size

    @staticmethod
    def _calc_sammon_cost(Y, D, inclusions = None):
        """ Calculate sammon stress.
        Y: map positions
        D: input distances

        No numba b/c cdist function is used.
        """

        if not inclusions is None:
            Y = Y[inclusions != 0, :]
            D = D[inclusions != 0, :][:, inclusions != 0]

        n_samples = Y.shape[0]
        Dinv = 1 / D
        D_map = cdist(Y,Y) + np.eye(n_samples)
        delta = D - D_map 
        E = ((delta**2)*Dinv).sum() 
        scale = 0.5 / (D.sum() - np.diag(D).sum())
        E = E * scale

        return E

    def _evaluate_cost_function(self, Ys, Ds, Ws, inclusions):
        """Evalute cost function

        Args:
            Ys (array): map coordiantes (n_samples, n_dims * n_periods)
            Ds (list): input distance matrices
            Ws (array): weights (n_samples, n_dims * n_periods)
            inclusions (array): (n_samples, n_periods)

        Returns:
            float, float, float: Cost function components
        """
        n_periods = len(Ds)
        cost_total = 0
        cost_temporal =  0
        costs_static = np.zeros(n_periods)
        for t in range(n_periods):
            Y_t = Ys[:,(self.n_dims*t):(self.n_dims*t)+self.n_dims]
            inc_t = inclusions[:,t]
            costs_static[t] = self._calc_sammon_cost(Y_t, Ds[t], inc_t)
        
        costs_static = np.sum(costs_static)
        cost_temporal = calc_dyn_cost(Ys, Ws, inclusions, self.p, self.n_dims)
        cost_total = costs_static + self.alpha * cost_temporal

        return cost_total, costs_static, cost_temporal 

    def _grad_descent(self, Xs, Ws, Y_inits, inclusions):
        """ Find minimum cost configuration via gradient descent. 
        For better convergence, we use a step halving procedure to identify
        appropriate step sizes."""

        n_dims = self.n_dims
        verbose = self.verbose
        alpha = self.alpha
        p = self.p
        max_iter = self.max_iter
        init_step_size = self.step_size

        # Initialize variables
        n_samples = Xs[0].shape[0]
        n_periods = len(Xs)
        D_inv_ts = []
        tol = self.tol
        last_cost = np.inf
        for t in range(n_periods):
            D_inv_ts.append(1/Xs[t])

        #-----------------------------------------------------#
        Ys = Y_inits                # Positions
        Ys_prev_iter = Ys.copy()
        dYs = np.zeros_like(Ys)     # Gradients
        #------------- Run Gradient Descent ---------------------#
        i = 0
        while i < max_iter:

            # Report cost values at start
            if i == 0:
                if verbose > 1:
                    cost_total, cost_static, cost_temporal = self._evaluate_cost_function(
                        Ys, Xs, Ws, inclusions)
                    print("[EvoSammon] Iteration {0} -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(i, cost_total, cost_static, cost_temporal))

            dY_temporals, _ = calc_dyn_gradient(
                Ys, 
                Ws, 
                inclusions, 
                p , 
                n_dims)

            for t in range(n_periods):

                Y_t = Ys[:,(n_dims*t):(n_dims*t)+n_dims]
                inc_t = inclusions[:,t]
                # Exclude non-included samples 
                Y_t = Y_t[inc_t != 0] 
                D_map_t = cdist(Y_t,Y_t) + np.eye(n_samples) # Use CDIST as its  faster than numba
                D_map_t_inv = 1. / D_map_t
                X_t = Xs[t]
                X_t = X_t[inc_t !=0,:][:, inc_t != 0]
                
                D_inv_t = D_inv_ts[t]
                D_inv_t = D_inv_t[inc_t !=0,:][:, inc_t != 0]

                dY_static = self._calc_gradient_sammon(Y_t, D_inv_t, D_map_t_inv )
                dY_static = np.reshape(dY_static, (-1,n_dims),order='F')

#               Y_t_old = Y_t
#               C_t = Cs[t]
#               step_size = self._get_step_size_sammon(
#                   init_step_size, Y_t_old, dY_static, X_t, D_inv_t, C_t, self.maxhalves)
                
                dY_static = extend_gradient(inc_t, dY_static, n_dims)

                # Calculate step size for static gradient
                # For temporal gradient, we keep a step size of 1 (might not yet be optimal to do so!)                
                dY_static = dY_static

                # Calculate full gradient
                dY_temporal_t = dY_temporals[t]
                
                # Static gradient is already negative
                dYs[:, (n_dims*t):(n_dims*t)+n_dims] = dY_static - alpha * dY_temporal_t

            # Note: For the first 5% of iterations, we use the initial step size    
            if i < max_iter / 20:
                step_size = self.step_size
            else:
                step_size = self._find_step_size_via_halving(Ys, dYs, Xs, Ws, inclusions)

            # Perform all updates
            Ys = Ys + step_size * dYs

            # ---- Report Progress ---- 
            check_iter = max_iter / 10

            # Check divergence 
            if np.any(np.isinf(Ys)):
                print("[EvoSammon] Divergent gradient detected. Iteration: {0}".format(i+1))
                raise DivergingGradientError()

            # Check convergence
            if np.linalg.norm(Ys - Ys_prev_iter) < tol:
                if verbose > 1:
                    print('[EvoSammon] Gradient norm vanished: Optimisation terminated')
                break

            # Report progress

            if (i+1)%check_iter == 0 or i == (max_iter-1):
                cost_total, cost_static, cost_temporal = self._evaluate_cost_function(
                    Ys, Xs, Ws, inclusions)

                if verbose > 1:
                    print("[EvoSammon] Iteration {0} -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(i+1, cost_total, cost_static, cost_temporal))
                if cost_total / last_cost > 1.01:
                    print("[EvoSammon] Iteration {0} -- Gradient starts diverging".format(i+1))
                    raise DivergingGradientError()
                last_cost = cost_total

            Ys_prev_iter[:,:] = Ys[:,:]
            i += 1
        cost_total, cost_static, cost_temporal = self._evaluate_cost_function(
            Ys, Xs, Ws, inclusions)
        final_positions = []
        for t in range(n_periods):
            final_positions.append(Ys[:, (n_dims*t):(t*n_dims)+n_dims])
            
        return final_positions, cost_total, cost_static, cost_temporal, Ws

    def fit_transform(self, Ds, inclusions = None):
        """Fit the distance matrices Ds and return embedded coordiantes. 

        Args:
            Ds (list): list of distance matrices (symmetric, non negative)
            inclusions (list, optional): list of inclusion vectors. Defaults to None.
        """
        if inclusions is None:
            inclusions = init_inclusions(Ds)
        
        n_samples = Ds[0].shape[0]
        for t in range(len(Ds)):
            Ds[t] = Ds[t] + np.eye(n_samples)     

        validate_input(Ds, inclusions)

        # For samming mapping, off-diagonal distances need to be strictly positive
        # to avoid division by zero
        for D in Ds:
            if np.count_nonzero(D<=0) > 0:
                raise ValueError("Off-diagonal dissimilarities must be strictly positive")   

        if self.verbose >= 1:
            print("[EvoSammon] -- Fitting EvoMap via SAMMON Mapping -- ")
       
        Y_inits = initialize_positions(
                Ds, 
                n_dims = self.n_dims, 
                Y_inits = self.init, 
                inclusions = inclusions,
                verbose = self.verbose)

        inclusion_array = build_inclusion_array(
            inclusions, 
            self.n_dims, 
            Y_inits)

        weights_array = calc_weights_new(Ds, self.n_dims, self.weighted)

        for ith_try in range(self.max_tries): 
            try:
                positions, cost_total, cost_static, cost_temporal, Ws = self._grad_descent(
                    Xs = Ds, 
                    Ws = weights_array,
                    Y_inits = Y_inits,
                    inclusions = inclusion_array)
                break
            except DivergingGradientError:
                print("[EvoSammon] Adjusting step sizes..")
                self.step_size /= 2
                Y_inits = initialize_positions(
                        Ds, 
                        n_dims = self.n_dims, 
                        Y_inits = None, 
                        inclusions = inclusions,
                        verbose = self.verbose)             
                continue

            if ith_try == self.max_tries -1:
                print("[EvoSammon] ERROR: Gradient descent failed to converge.")
                return -1


        self.Y_ts_ = positions
        self.cost_total_ = cost_total
        self.cost_static_avg_ = cost_static / len(Ds)
        self.cost_temporal_ = cost_temporal
        self.Ws_ = Ws
        return self.Y_ts_

    def grid_search(self, param_grid, X_ts):

        if self.verbose > 0:
            print("[EvoSammon] Evaluating parameter grid..")

        def iter_grid(p):
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params
            
        df_res = pd.DataFrame(columns = ['alpha', 'p', 'avg_hitrate', 'misalign', 'align', 'pers'])
        for param_combi in iter_grid(param_grid):
            if self.verbose > 0:
                print("[EvoSammon] .. evaluating parameter combination: " + str(param_combi))
            model_i = copy.deepcopy(self)
            model_i.set_params(param_combi)
            model_i.set_params({'verbose': 0})
            Y_ts_i = model_i.fit_transform(X_ts)
            df_res = df_res.append({
                'alpha': param_combi['alpha'], 
                'p': int(param_combi['p']), 
                'static_cost': model_i.cost_static_avg_,
                'avg_hitrate': avg_hitrate_score(X_ts, Y_ts_i, n_neighbors= 3, input_type = dict(model_i.get_params())['input_type']),
                'misalign': misalign_score(Y_ts_i),
                'misalign_norm': misalign_score(Y_ts_i, normalize = True),
                'align': align_score(Y_ts_i), 
                'pers': persistence_score(Y_ts_i)}, ignore_index = True)

        if self.verbose > 0:
                print("[EvoSammon] Done.")


        return df_res