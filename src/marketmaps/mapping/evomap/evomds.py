COORD_SYS_BOUND = 1e2
from matplotlib.pyplot import step
from numpy.lib.function_base import disp
#from sklearn.metrics.pairwise import cosine_similarity
from ._core import calc_dyn_gradient
from ._core import initialize_positions
from ._core import build_inclusion_array
from ._core import calc_weights
from ._core import calc_weights_new
from ._core import calc_dyn_cost

from scipy.spatial.distance import cdist
from scipy.linalg import norm
#from sklearn.isotonic import IsotonicRegression
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import itertools
import random
from itertools import product
import copy

import numpy as np
from numba import jit

from scipy.linalg import norm

class Error(Exception):
    """Base class for other exceptions"""
    pass

class DivergingGradientError(Error):
    """Raised when the input value is too small"""
    pass

@jit(nopython=True)
def mean_squared_error(expected_value, actual_value):
    return pow(expected_value-actual_value, 2)

@jit(nopython=True)
def compute_exact_gradient(positions, distances, disparities):
    n_samples = distances.shape[0]
    n_dims = positions.shape[1]
    gradient = np.zeros(shape = (n_samples, n_dims))
    for i in range(n_samples):
        for l in range(n_dims):
            grad_il = 0
            for j in range(n_samples):
                if j != i:
                    grad_il += (1- disparities[i,j] / distances[i,j]) * (positions[j,l] - positions[i,l])
            gradient[i,l] = grad_il
    return gradient / (n_samples - 1)        

@jit(nopython=True)
def normalized_stress(disparities, positions):
    stress = 0
    sum_dist = 0
    for i in range(len(disparities)):
        for j in range(len(disparities[0])):
            if i < j:
                dist_ij = dist(positions[i], positions[j])
                stress += mean_squared_error(disparities[i][j], dist_ij)
                sum_dist += dist_ij**2

    stress /= sum_dist
    stress = np.sqrt(stress)
    return stress        

@jit(nopython=True)
def dist(pos1, pos2):
    """
    Compute the Euclidean distance between two locations (numpy arrays) a and b
    Thus, dist(pos[1], pos[2]) gives the distance between the locations for items 1 and 2

    @param pos1: Position 1
    @param pos2: Position 2
    @return Returns the Euclidean distance between these two positions. 

    This function is not the most efficient, but the data we are working with is rather small.
    Also, this makes the code much more readable!

    >>> dist([0, 0], [3, 4])
    5.0
    >>> dist([1, 2, 3], [3, 4 ,4])
    3.0
    """
    differences = [pos1[i] - pos2[i] for i in range(len(pos1))]
    squared_diff = np.array([val*val for val in differences])
    return np.sqrt(np.sum(squared_diff))


class EvoMDS():
    """ EvoMap implemented for (metric) Multidimensional Scaling."""
    
    def __init__(
        self, 
        alpha = 0,
        p = 1,
        weighted = 'exponential',
        n_dims = 2,
        metric = True, 
        max_iter = 2000, 
        init = None, 
        verbose = 0, 
        optim = 'GD',
        input_type = 'distance',
        maxhalves = 10,
        tol = 1e-3, 
        n_inits = 1,
        step_size = .1, 
        max_tries = 10):
                
        self.alpha = alpha
        self.p = p
        self.weighted = weighted
        self.maxhalves = maxhalves
        self.optim = optim
        self.n_dims = n_dims
        self.max_iter = max_iter
        self.input_type = input_type
        self.metric = metric
        self.step_size = step_size
        self.max_tries = max_tries
        
        if not init is None: 
            self.init = init.copy()
        else:
            self.init = init
            
        self.verbose = verbose
        self.tol = tol
        self.n_inits = n_inits

    def get_params(self):
        return self.__dict__.items()

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

        
    def _validate_input(self, Xs, inclusions):
        if len(Xs) != len(inclusions):
            raise ValueError("Unequal number of distance matrices and inclusions.")
            
        if not Xs[0].shape[0] == inclusions[0].shape[0]:
            raise ValueError("Inclusions do not match distance matrices.")

    def _get_mag(self, gradient, positions):
        if not np.isfinite(gradient).all():
            raise ValueError("Gradient diverging")
            
        if not np.isfinite(positions).all():
            raise ValueError("Positions diverging")
            
        mag = norm(gradient)/norm(positions)
        return mag        
               

    def grid_search(self, param_grid, X_ts):

        if self.verbose > 0:
            print("[EvoMDS] Evaluating parameter grid..")

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
                print("[EvoMDS] .. evaluating parameter combination: " + str(param_combi))
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
                print("[EvoMDS] Done.")


        return df_res


    def _calc_rank_corr(self, Ys, Xs):
        
        n_periods = len(Xs)
        avg_rank_corr = 0
        for t in range(n_periods):

            dist_map = np.tril(squareform(pdist(Ys[:, (2*t):(t*2)+2]))).ravel()
            dist_data = np.tril(Xs[t]).ravel()
            # Build matrix of pairwise labels and ravel it flat similar to distances
            indices = dist_map != 0
            dist_map = dist_map[indices]
            dist_data = dist_data[indices]

            indices = np.argsort(dist_data)
            dist_map_sorted = dist_map[indices]
            dist_data_sorted = dist_data[indices]

            avg_rank_corr += spearmanr(dist_map_sorted, dist_data_sorted)[0]
        
        avg_rank_corr /= n_periods
        avg_rank_corr = np.round(avg_rank_corr, 2)
        return avg_rank_corr
    
        

    def _get_step_size(self, current_stress, min_stress, delta):
        return current_stress*delta/min_stress
    
    def _find_step_size(self, Disp, Y_old, dY):

        step_size = self.step_size
        E_old = normalized_stress(Disp, Y_old)
        for j in range(self.maxhalves):
            Y = Y_old + step_size * dY
            E_new = normalized_stress(Disp, Y)
            if E_new < E_old:
                break
            else:
                step_size = 0.5* step_size
        return step_size

    def _find_step_size_via_halving(self, Ys, dYs, disparities, Ws, inclusions):

        C_old,_,_ = self._evaluate_cost_function(Ys, disparities, Ws, inclusions)
        step_size = self.step_size
        for j in range(self.maxhalves):
            Ys_new = Ys + step_size * dYs
            C_new,_,_ = self._evaluate_cost_function(Ys_new, disparities, Ws, inclusions)
            if C_new < C_old:
                break
            else:
                step_size = 0.5 * step_size
            
        return step_size


    def _get_avg_stress(self, Xs, Ys, inclusions, return_all_periods = False):
        """[summary]

        Args:
            Xs (array): disparities (n_samples, n_samples, n_periods)
            Ys (array): map coordinates (n_samples, n_dims * n_periods)
            inclusions (array): inclusions (n_samples, n_periods)
            return_all_periods (bool, optional): Return full series?. Defaults to False.

        Returns:
            [type]: [description]
        """

        avg_stress = 0
        n_periods = Xs.shape[2]
        n_dims = self.n_dims
        stress_ts = []
        for t in range(n_periods):
            # Pick all positions at time t which are included:
            Y_t = Ys[:, (n_dims*t):(n_dims*t)+n_dims]
            inc_t = inclusions[:,t]

            Y_t = Y_t[inc_t != 0]
            if not Xs.shape[0] == Y_t.shape[0]:
                X_t = Xs[:,:, t][inc_t != 0, :][:, inc_t != 0]
            else:
                X_t = Xs[:,:, t]

            stress_t = normalized_stress(X_t, Y_t)
            stress_t = np.round(stress_t,4)
            avg_stress += stress_t
            stress_ts.append(stress_t)
        avg_stress /= n_periods

        if return_all_periods:
            return avg_stress, stress_ts

        else:
            return avg_stress

    def _calc_alignment(self, Ys, inclusions, n_dims = 2):

        algn = 0
        n_periods = Ys.shape[1] / n_dims
        n_periods = int(n_periods)
        if n_periods == 1:
            return 0 

        for t in range(1,n_periods):
            Y_this = Ys[:, (2*t):(t*2)+2]
            Y_prev = Ys[:, (2*(t-1)):((t-1)*2)+2]
            inc = (inclusions[:,t] != 0)*(inclusions[:,t-1] != 0)
            Y_this = Y_this[inc]
            Y_prev = Y_prev[inc]
            
            algn += np.diag(cosine_similarity(Y_this, Y_prev)).mean()
        
        algn /= (n_periods -1)
        return np.round(algn, 2)
    
    
    def _extend_gradient(self, Y_t, inc_t, dY_static):
        """If some observations were excluded from the gradient computations, add
        0s for their gradients now."""
        
        n_samples = len(inc_t)
        if Y_t.shape[0] < n_samples:
            dY_static_full = np.zeros((n_samples, self.n_dims))
            k = 0
            for j in range(n_samples):
                if inc_t[j] != 0:
                    dY_static_full[j, :] = dY_static[k, :]
                    k += 1
        else:    
            dY_static_full = dY_static
            
        return dY_static_full


    def _evaluate_cost_function(self, Ys, Ds, Ws, inclusions):
        """Evalute cost function

        Args:
            Ys (array): map coordiantes (n_samples, n_dims * n_periods)
            Ds (array): disparities (n_samples , n_samples, n_periods)
            Ws (array): weights (n_samples, n_dims * n_periods)
            inclusions (array): (n_samples, n_periods)

        Returns:
            float, float, float: Cost function components
        """
        cost_total = 0
        cost_temporal =  0
        _, stress_ts = self._get_avg_stress(Ds, Ys, inclusions, return_all_periods = True)
        cost_static = np.sum(stress_ts)
        cost_temporal = calc_dyn_cost(Ys, Ws, inclusions, self.p, self.n_dims)
        cost_total = cost_static + self.alpha * cost_temporal

        return cost_total, cost_static, cost_temporal 

    def _calc_decay(self, mu_start, mu_end, steps_max):
        """Solve for lambda of an exponential decay function. 

        see: Graph Drawing by Stochastic Gradient Descent (Zheng, Pawar, Goodman)

        """
        lamb = np.log(mu_end/mu_start) / - steps_max
        return lamb

    def _calc_decayed_step_size(self, lamb, mu_start, iteration):
        step = mu_start * np.exp(-lamb * iteration)
        return step

    @staticmethod
    @jit(nopython=True)
    def _perform_one_SGD_step(pairs, n_periods, n_dims, inclusions, disparities, Ys, stepsize, alpha, p, Ws):
        
        for i in range(pairs.shape[0]):
            idx_i = pairs[i, 0]
            idx_j = pairs[i, 1]
            for t in range(n_periods):
                # Check if both included in t
                if (inclusions[idx_i, t] * inclusions[idx_j, t] == 1):

                    Y_t = Ys[:,(n_dims*t):(n_dims*t)+n_dims] 
                    yi = Y_t[idx_i,:]
                    yj = Y_t[idx_j,:]
                    dij = disparities[idx_i, idx_j, t]

                    dist_ij = np.linalg.norm(yi-yj)
                    delta = (dist_ij - dij) / dist_ij * (yi-yj)

                    mu_t = np.minimum(1, stepsize)
                    step_i = - mu_t / 2 * delta
                    step_j = mu_t / 2 * delta

                    # Perform the two updates
                    indices = np.array([idx_i, idx_j])
                    Ys[idx_i, (n_dims*t):(n_dims*t)+n_dims] = Ys[idx_i, (n_dims*t):(n_dims*t)+n_dims] + step_i
                    Ys[idx_j, (n_dims*t):(n_dims*t)+n_dims] = Ys[idx_j, (n_dims*t):(n_dims*t)+n_dims] + step_j

                    # Finally, add temporal component
                    dY_temporals, _ = calc_dyn_gradient(Ys[indices, :], Ws[indices, :], inclusions[indices,:], p, n_dims)
                    Ys[idx_i, (n_dims*t):(n_dims*t)+n_dims] -= alpha * dY_temporals[t][0, :]
                    Ys[idx_j, (n_dims*t):(n_dims*t)+n_dims] -= alpha * dY_temporals[t][1, :]
                else:
                    continue

        return Ys

    def _grad_descent(self, Xs, Ws, Y_inits, inclusions):
        """ A single optimization via gradient descent. """
        
        n_dims = self.n_dims
        max_iter = self.max_iter
        metric = self.metric
        tol = self.tol
        verbose = self.verbose
        alpha = self.alpha
        p = self.p
        step_size = self.step_size

        # Initialize variables
        n_samples = Xs[0].shape[0]
        n_periods = len(Xs)

        # Initialize gradient descent
        i = 0

        #------------- Initialize Optimization Parameters ---------------------#
        Ys = Y_inits.copy()                     # Positions
        dYs = np.zeros_like(Ys)          # Gradients (direction)
        Ys_prev_iter = Ys.copy()
        last_cost = np.inf
        if self.optim == 'SGD':
            # Calculate decay function for step size
            mu_start = self.step_size
            mu_end = 0.01
            lamb = self._calc_decay(mu_start, mu_end, max_iter)

        #------------- Run Gradient Descent ---------------------#
        while i < max_iter:            
            if i == 0:
                # Calculate initial stress (before first update)
                disparities = np.zeros((n_samples, n_samples, n_periods))
                for t in range(n_periods):
                    X_t = Xs[t]
                    if metric: 
                        disparities_t = X_t
                    else:
                        Y_t = Ys[:,(n_dims*t):(n_dims*t)+n_dims] 
                        dis = cdist(Y_t, Y_t)
                        disparities_t = self._transform_distances(D = X_t, dis = dis)
                        
                    disparities[:,:,t] = disparities_t

                # report cost values at start:
                if verbose > 1:
                    cost_total, cost_static, cost_temporal = self._evaluate_cost_function(
                        Ys, disparities, Ws, inclusions)
                    print("[EvoMDS] Iteration {0} -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(i, cost_total, cost_static, cost_temporal))


            # ---------------------------------------------------------------------
            if self.optim == 'GD':
                # Perform one iteration of gradient descent
                dY_temporals, _ = calc_dyn_gradient(Ys, Ws, inclusions, p, n_dims)
                disparities = np.zeros((n_samples, n_samples, n_periods))
                for t in range(n_periods):
                    # Exclude non-included samples for gradient calculations
                    Y_t = Ys[:,(n_dims*t):(n_dims*t)+n_dims]
                    inc_t = inclusions[:, t]
                    X_t = Xs[t]
                    X_t = X_t[inc_t != 0,:][:, inc_t != 0]
                    Y_t = Y_t[inc_t != 0] 
                    
                    # Calculate map distances
                    dis = cdist(Y_t, Y_t)

                    if metric: 
                        disparities_t = X_t
                    else:
                        # Non-Metric: Transform dissimilarities (data) to disparities
                        # Get upper triangle and drop zeros
                        disparities_t = self._transform_distances(D = X_t, dis = dis)
                        
                    disparities[:, :, t] = disparities_t

                    dY_static = compute_exact_gradient(Y_t, dis, disparities_t)
                    # Add zeros for gradient of non-included samples:
                    dY_static = self._extend_gradient(Y_t, inc_t, dY_static)
                    
                    # Calculate full gradient
                    # Static gradient is already negative (i.e. right direction)
                    # Temporal gradient points uphill -> *-1
                    dYs[:, (n_dims*t):(n_dims*t)+n_dims] = dY_static - self.alpha * dY_temporals[t]
            
                # Note: For the first 5% of iterations, we use the initial step size    
                if i < max_iter / 20:
                    step_size = self.step_size
                else:
                    step_size = self._find_step_size_via_halving(Ys, dYs, disparities, Ws, inclusions)

                # Perform all updates
                Ys = Ys + step_size * dYs

                # --- END OF ONE STEP CLASSIC GD --- 

            elif self.optim == 'SGD':
                stepsize = self._calc_decayed_step_size(lamb, mu_start, i)
                pairs = np.array(list(itertools.combinations(range(n_samples),2)))
                random.shuffle(pairs)
                disparities = np.zeros((n_samples, n_samples, n_periods))
                for t in range(n_periods):
                    if metric: 
                        disparities_t = X_t
                    else:
                        # Calculate map distances and transform disparities via isotonic regression
                        dis = cdist(Y_t, Y_t)
                        disparities_t = self._transform_distances(D = X_t, dis = dis)
                    disparities[:,:,t] = disparities_t

                
                Ys = self._perform_one_SGD_step(pairs, n_periods, n_dims, inclusions,disparities, Ys, stepsize, alpha, p, Ws)
            else:
                raise ValueError("Unknown optimization routine.")

            # Check divergence 
            if np.any(Ys > 1/tol):
                print("[EvoMDS] Divergent gradient detected. Iteration: {0}".format(i+1))
                raise DivergingGradientError()

            # ---- Report Progress ---- 
            check_iter = max_iter / 10

            # ----- Check Convergence After Udpates -----
            if np.linalg.norm(Ys - Ys_prev_iter) < tol:
                if verbose > 1:
                    cost_total, cost_static, cost_temporal = self._evaluate_cost_function(
                        Ys, disparities, Ws, inclusions)
                    print('[EvoMDS] Iteration {0} -- Gradient norm vanished: Optimisation terminated'.format(i+1))
                    print("[EvoMDS] Iteration {0} -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(i+1, cost_total, cost_static, cost_temporal))

                break
            elif i == max_iter-1:
                if verbose > 1:
                    print('[EvoMDS] Maximum number of iterations exceeded.')

            if (i+1)%check_iter == 0 or i == (max_iter-1):
                cost_total, cost_static, cost_temporal = self._evaluate_cost_function(
                    Ys, disparities, Ws, inclusions)

                if verbose > 1:
                    print("[EvoMDS] Iteration {0} -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(i+1, cost_total, cost_static, cost_temporal))
                if cost_total / last_cost > 1.01:
                    print("[EvoMDS] Iteration {0} -- Gradient starts diverging".format(i+1))
                    raise DivergingGradientError()
                last_cost = cost_total

            Ys_prev_iter[:,:] = Ys[:,:]
            i = i +1
       
        final_avg_stress, final_stress_ts = self._get_avg_stress(
            disparities, Ys, inclusions, return_all_periods= True
        ) 

        cost_total, cost_static, cost_temporal = self._evaluate_cost_function(
            Ys, disparities, Ws, inclusions)


        # Re-Transform arrays to list for easier access
        final_positions = []
        for t in range(n_periods):
            final_positions.append(Ys[:, (n_dims*t):(t*n_dims)+n_dims])
            
        return final_positions, cost_total, cost_static, cost_temporal, Ws
       
        
    def fit_transform(self, Xs, inclusions = None):
        """
        Fit the data from D and return the embedding coordinates.
        """        

        n_periods = len(Xs)
        Ds = []
        if self.input_type == 'vector':
            for t in range(n_periods):
                Ds.append(cdist(Xs[t], Xs[t]))

        elif self.input_type == 'distance':
            Ds = Xs.copy()

        else:
            raise ValueError("Unknown input type. Should be 'vector' or 'distance'.")


        # Create inclusion arrays if none provided
        if inclusions is None:
            inclusions = []
            for t in range(len(Ds)):
                inclusions.append(np.ones(Ds[0].shape[0]))


        self._validate_input(Ds, inclusions)
        if self.verbose > 0:
            if self.metric:
                print("[EvoMDS] -- Fitting EvoMap via Metric MDS -- ")
            else:
                print("[EvoMDS] -- Fitting EvoMap via Non-Metric MDS -- ")

        best_pos = None
        best_stress = np.inf

        weights_array = calc_weights_new(Ds, self.n_dims, self.weighted)
        if self.init is None:
            
            for i in range(self.n_inits):
                
                if self.verbose >= 1:
                    print("[EvoMap] -- Using random initialization")
                    print("[EvoMap] -- MDS iteration {0}/{1}".format(i+1, self.n_inits))

                Y_inits = initialize_positions(
                    Ds, 
                    n_dims = self.n_dims, 
                    Y_inits = self.init, 
                    inclusions = inclusions)

                inclusion_array = build_inclusion_array(inclusions, self.n_dims, Y_inits)
                for ith_try in range(self.max_tries):
                    try:
                        positions, cost_total, cost_static, cost_temporal, Ws = self._grad_descent(
                            Xs = Ds, 
                            Ws = weights_array, 
                            Y_inits = Y_inits, 
                            inclusions = inclusion_array)
                        break
                    except DivergingGradientError:
                        print("[EvoMDS] Adjusting step sizes..")
                        self.step_size /= 2
                        # Potential idea: Re-initialize the positions!
#                       Y_inits = initialize_positions(
#                           Ds, 
#                           n_dims = self.n_dims, 
#                           Y_inits = None, 
#                           inclusions = inclusions)

                    if ith_try == self.max_tries -1:
                        print("[EvoMDS] ERROR: Gradient descent failed to converge.")
                        return -1

                avg_stress = cost_static / len(Ds)
                if avg_stress < best_stress:
                        best_pos = positions
                        best_stress = avg_stress

        else:
            if self.verbose >= 1:
                print("[EvoMap] -- Using pre-initialized values")
            Y_inits = initialize_positions(Ds, n_dims = self.n_dims, Y_inits = self.init, inclusions = inclusions)            
            inclusion_array = build_inclusion_array(inclusions, self.n_dims, Y_inits)
            for ith_try in range(self.max_tries):
                try:
                    best_pos, cost_total, cost_static, cost_temporal, Ws = self._grad_descent(
                        Xs = Ds, 
                        Ws = weights_array, 
                        Y_inits = Y_inits,
                        inclusions = inclusion_array)
                    break
                except DivergingGradientError:
                    print("[EvoMDS] Adjusting step sizes..")
                    self.step_size /= 2
#                   Y_inits = initialize_positions(
#                       Ds, 
#                       n_dims = self.n_dims, 
#                       Y_inits = None, 
#                       inclusions = inclusions)
                                        
                if ith_try == self.max_tries -1:
                    print("[EvoMDS] ERROR: Gradient descent failed to converge.")
                    return -1

        self.Ws_ = Ws
        self.Y_ts_ = best_pos 
        self.cost_total_ = cost_total
        self.cost_temporal_ = cost_temporal
        self.cost_static_avg_ = cost_static / len(Ds)
        self.cost_static_ = cost_static
        return self.Y_ts_