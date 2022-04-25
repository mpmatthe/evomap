""" EvoMap implemented for t-SNE.

Author: Maximilian Peter Matthe.
This version: December 2021.
"""
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

import numpy as np
import pandas as pd
from scipy import linalg
from time import time
import sys
import os
import copy
from itertools import product 

from ._core import calc_dyn_cost, calc_p_matrix
from ._core import initialize_positions
from ._core import build_inclusion_array
from ._core import calc_q_matrix
from ._core import calc_gradient
from ._core import calc_dyn_gradient
from ._core import is_valid_input
from ._core import kl_divergence
from ._core import calc_weights
from ._core import calc_weights_new
from ._core import grid_search

class Error(Exception):
    """Base class for other exceptions"""
    pass

class DivergingGradientError(Error):
    """Raised when the input value is too small"""
    pass

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class EvoTSNE():
    """ EvoMa implemented for t-distributed stochastic neighborhood embedding (TSNE)"""
    
    def __init__(
        self,  
        alpha = 0, 
        p = 1, 
        weighted = 'exponential', 
        n_dims=2, 
        perplexity=15, 
        max_iter = 2000, 
        stop_lying_iter = 250, 
        input_type = "distance",
        early_exaggeration = 4, 
        eta = 'auto', 
        init = None, 
        verbose = 0,
        max_tries = 5, 
        tol = 1e-5):

        self.alpha = alpha
        self.p = p
        self.weighted = weighted
        self.n_dims = n_dims
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.stop_lying_iter = stop_lying_iter
        self.input_type = input_type
        self.early_exaggeration = early_exaggeration
        self.eta = eta 
        self.init= init 
        self.verbose = verbose
        self.max_tries = max_tries
        self.tol = tol

        self.grid_search = grid_search

    def get_params(self):
        return self.__dict__.items()

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        return self


    def _evaluate_cost_function(self, Ys, Ps, Qs, Ws, inclusions):
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
        cost_static = 0
        n_periods = len(Ps)
        for t in range(n_periods):
            C = kl_divergence(Ps[t],Qs[t],inclusions[:, t])
            cost_static += C

        cost_temporal = calc_dyn_cost(Ys, Ws, inclusions, self.p, self.n_dims)
        cost_total = cost_static + self.alpha * cost_temporal

        return cost_total, cost_static, cost_temporal 

    def fit_transform(self,Xs, inclusions = None):
        """ Fit a low dimensional embedding to the data in X. 

        Args:
            Xs (list): Time indexed input data.
            inclusions ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]

        Returns:
            list: List of map coordinates (arrays)
        """

        warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

        #  ------- Check Input Data ------- #
        n_periods = len(Xs)
        for t in range(n_periods):
            if not is_valid_input(Xs[t], self.input_type):
                raise ValueError("Invalid input data.")

        if self.p >= len(Xs) and self.alpha > 0:
            raise ValueError("Order of Distance needs to be lower than the number of periods.")
        
        #  ------- Initialize Variables ------- #
        (n, d) = Xs[0].shape    
        if self.verbose > 0:
            print("[EvoTSNE] -- Fitting EvoMap via TSNE -- ")

        # Create inclusion lists if none provided
        if inclusions is None:
            inclusions = []
            for t in range(n_periods):
                inclusions.append(np.ones(n))

        # Initialize solution
        Y_inits = initialize_positions(
            Xs, 
            n_dims = self.n_dims, 
            Y_inits = self.init, 
            inclusions = inclusions)

        # Transform list to array form
        inclusion_array = build_inclusion_array(inclusions, self.n_dims, Y_inits)

        # Get P-Matrices
        if self.verbose > 1:
            print("[EvoTSNE] Calculating P matrices of t-SNE...")
        Ps = []
        for t in range(n_periods):
            P = calc_p_matrix(
                Xs[t], 
                inclusions[t], 
                self.input_type, 
                self.perplexity)
            Ps.append(P)

        # Calculate weights (based on P matrices)

        # TODO: After testing all weight schemes, simplify here
        if self.weighted == None:
            if self.verbose > 0 and self.alpha > 0:
                print("[EvoTSNE] Applying no weights")
            Ws = np.ones((n, self.n_dims*n_periods))

        elif self.weighted == "distances": 
            if self.input_type == "distance":
                Ws = calc_weights(Xs, self.n_dims, weighting_scheme = "relations")
            else: 
                raise ValueError("Distance weights can only be applied for input type 'distances'")

        elif self.weighted == "relations":
            Ws = calc_weights(Ps, self.n_dims, weighting_scheme = "relations")

        elif self.weighted == "neighbors":
            Ws = calc_weights(Ps, self.n_dims, weighting_scheme = "neighbors")
        
        elif self.weighted == "nearest_relations":
            Ws = calc_weights(Ps, self.n_dims, weighting_scheme = "nearest_relations")
        
        elif self.weighted == "nearest_kl_divergence":
            Ws = calc_weights(Ps, self.n_dims,  weighting_scheme = "nearest_kl_divergence")

        elif self.weighted == "kl_divergence":
            Ws = calc_weights(Ps, self.n_dims, weighting_scheme = "kl_divergence")  

        # -- New Weights -- 
        elif self.weighted == "inverse":
            Ws =  calc_weights_new(Xs, self.n_dims, weighting_scheme= 'inverse')

        elif self.weighted == "inverse_plus":
            Ws = calc_weights_new(Xs, self.n_dims, weighting_scheme = 'inverse_plus')
            
        elif self.weighted == "mirror":
            Ws = calc_weights_new(Xs, self.n_dims, weighting_scheme = 'mirror')
        
        elif self.weighted == "exponential":
            Ws = calc_weights_new(Xs, self.n_dims, weighting_scheme = 'exponential')

        else: raise ValueError("Weights should be one of [None, 'relations', 'neighbors', 'nearest_relations'], not " + str(self.weighted))

        for t,P in enumerate(Ps):
            assert np.all(np.isfinite(P)), "All probabilities should be finite. Check if object at index {0} at time {1} has sufficient non-zero neighbors.".format(np.where(np.isfinite(P) == False)[0][0], t)
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), ("All probabilities should be less "
                                        "or then equal to one")

        if self.eta == "auto":
            # See issue #18018
            self.eta = Xs[0].shape[0] / self.early_exaggeration
            self.eta = np.maximum(self.eta, 50)
        else:
            if not (self.eta > 0):
                raise ValueError("learning_rate 'eta' must be a positive number or 'auto'.")

        # ---- Run Gradient Descent: -----  
        for ith_try in range(self.max_tries):
            try: 
                final_positions, cost_total, cost_static, cost_temporal, Ws = self.grad_descent_evo_tsne(Ps, Ws, Y_inits, inclusion_array)
                break
            except DivergingGradientError:
                if self.verbose > 0:
                    print("[EvoTSNE] Adjusting step sizes...")
                self.eta = self.eta/10
            
            if ith_try == self.max_tries -1:
                print("[EvoTSNE] ERROR: EvoMap failed to converge.")
                return -1
        
        self.Ws_ = Ws
        self.Y_ts_ = final_positions 
        self.cost_total_ = cost_total
        self.cost_temporal_ = cost_temporal
        self.cost_static_avg_ = cost_static / len(Xs)       

        return final_positions

    def grad_descent_evo_tsne(self, Ps, Ws, Y_inits, inclusions):
        """ Optimize EvoTSNE via momentum-based gradient descent"""

        #  ------- Optimization Parameters ------- #

        initial_momentum = 0.5
        final_momentum = 0.8 
        min_gain = 0.01
        mom_switch_iter = 250
        n_iter_check = 250
        tol = self.tol
        eta = self.eta
        alpha = self.alpha
        verbose = self.verbose
        early_exaggeration = self.early_exaggeration
        max_iter = self.max_iter
        p = self.p
        verbose = self.verbose
        np.seterr(over='raise')

        #------------- Initialize Optimization Variables ---------------------#

        Ys =  Y_inits.copy()                     # Positions
        dYs = np.zeros_like(Ys)          # Gradients (direction)
        iYs = np.zeros_like(Ys)          # Step sizes
        gains = np.ones_like(Ys)
        Ys_prev_iter = Ys.copy()

        Ps = Ps.copy()
        n_periods = len(Ps)
        n_dims = self.n_dims

        # early exaggeration
        for t in range(n_periods):
            Ps[t] = Ps[t] * early_exaggeration
        momentum = initial_momentum

        #------------- Run Gradient Descent ---------------------#
        for iter in range(max_iter):


            dY_temporals, _ = calc_dyn_gradient(
                Ys, Ws, inclusions, p, n_dims)

            Qs = []
            # Calculate all static gradients (along full time-series)
            for t in range(n_periods):
                Y_t = Ys[:,(n_dims*t):(n_dims*t)+n_dims]
                inc_t = inclusions[:, t]
                
                Q_t, dist_t = calc_q_matrix(Y_t, inc_t)
                Qs.append(Q_t)
            
                dY_t_static = calc_gradient(Y_t, Ps[t], Q_t, dist_t)

                # Calculate full gradient (pointing uphill)
                dYs[:, (n_dims*t):(n_dims*t)+n_dims] = dY_t_static + (alpha * dY_temporals[t])

                # Calculate momentum gains (based on original TSNE implementation)
                iY_t = iYs[:, (n_dims*t):(n_dims*t)+n_dims]
                dY_t = dYs[:, (n_dims*t):(t*n_dims)+n_dims]
                dec = np.sign(iY_t) == np.sign(dY_t)
                inc = np.invert(dec)

                gains[:, (n_dims*t):(t*n_dims)+n_dims][inc] += .2
                gains[:, (n_dims*t):(t*n_dims)+n_dims][dec] *= .8

            # report cost values at start:
            if iter == 0 and verbose > 1:
                cost_total, cost_static, cost_temporal = self._evaluate_cost_function(Ys, Ps, Qs, Ws, inclusions)
                print("[EvoTSNE] Iteration {0} -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(iter+1, cost_total, cost_static, cost_temporal))

            np.clip(gains, min_gain, np.inf, out=gains)
            # OPTIONAL? Clip gradient norm to avoid exploding gradients
            #dYs = dYs / np.linalg.norm(dYs)

            # ------ Perform all updates ------ 
            iYs = momentum * iYs - eta * (gains * dYs)
            Ys = Ys + iYs 

            # ------ Check divergence ------  
            if np.any(Ys > 1/tol):
                if verbose > 0:
                    print("[EvoTSNE] Divergent gradient at iteration {0}".format(iter+1))
                raise DivergingGradientError()

            # ----- Check Convergence After Udpates -----
            if np.linalg.norm(Ys - Ys_prev_iter) < tol:
                if verbose > 1:
                    print('[EvoTSNE] Iteration {0} -- Gradient norm vanished: Optimisation terminated'.format(iter+1))

                break
            elif iter == max_iter-1:
                if verbose > 1:
                    print('[EvoTSNE] Maximum number of iterations exceeded.')

            # Store current solution (note: need to use indexes to copy vlaues, 
            # rather than setting a pointer)
            Ys_prev_iter[:,:] = Ys[:,:]

            # Switch Momentum if necessary
            if iter == mom_switch_iter:
                momentum = final_momentum

            # Readjust early exaggeration
            if iter == self.stop_lying_iter:
                for t in range(n_periods):
                    Ps[t] = Ps[t] / early_exaggeration

            # Compute current value of cost function
            report_progress = (iter +1) % n_iter_check == 0
            if report_progress and verbose > 1 and iter < (max_iter-1):         
                cost_total, cost_static, cost_temporal = self._evaluate_cost_function(Ys, Ps, Qs, Ws, inclusions)
                print("[EvoTSNE] Iteration {0} -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(iter+1, cost_total, cost_static, cost_temporal))
                
        # Compute final cost values
        cost_total, cost_static, cost_temporal = self._evaluate_cost_function(Ys, Ps, Qs, Ws, inclusions)
        if verbose > 0:
            print("[EvoTSNE] Final positions: -- Cost: {1:.2f} -- Static: {2:.2f} -- Temp.: {3:.2f}".format(iter+1, cost_total, cost_static, cost_temporal))

        final_positions = []
        for t in range(n_periods):
            final_positions.append(Ys[:, (n_dims*t):(t*n_dims)+n_dims])

        return final_positions, cost_total, cost_static, cost_temporal, Ws
