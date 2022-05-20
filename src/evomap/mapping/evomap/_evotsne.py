import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from ._core import EvoMap
from ._core import _evomap_cost_function, _get_positions_for_period
from evomap.mapping._core import DivergingGradientError

class EvoTSNE(EvoMap):

    def __init__(
        self,
        alpha = 0,                      
        p = 1,                          
        weighting = 'exponential',       
        n_dims = 2, 
        perplexity = 15,  
        stop_lying_iter = 250,
        early_exaggeration = 4,
        eta = 'auto',
        initial_momentum = .5,
        final_momentum = .8,
        n_iter = 2000,  
        n_iter_check = 50,
        init = None, 
        verbose = 0, 
        input_type = 'distance', 
        maxhalves = 5, 
        tol = 1e-3,  
        n_inits = 1, 
        max_tries = 5
    ):

        super().__init__(alpha = alpha, p = p, weighting= weighting)
        self.n_dims = n_dims
        self.perplexity = perplexity
        self.stop_lying_iter = stop_lying_iter
        self.early_exaggeration = early_exaggeration
        self.eta = eta
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.n_iter = n_iter
        self.n_iter_check = n_iter_check
        self.init = init
        self.verbose = verbose
        self.input_type = input_type
        self.maxhalves = maxhalves
        self.tol = tol
        self.n_inits = n_inits
        self.max_tries = max_tries
        self.method_str = "EvoTSNE"

    def fit(self, Xs):
        self.fit_transform(Xs)
        return self

    def fit_transform(self, Xs):
        from evomap.mapping._core import gradient_descent_with_momentum
        from evomap.mapping._tsne import _kl_divergence, _check_prepare_tsne

        super()._validate_input(Xs)
                
        # Check and prepare input data
        n_periods = len(Xs)
        if self.input_type == 'distance':
            Ds = Xs
        elif self.input_type == 'vector':
            Ds = []
            for t in range(n_periods):
                Ds.append(cdist(Xs[t],Xs[t]))
        else:
            raise ValueError("Input type should be 'distance' or 'vector', not {0}".format(self.input_type))

        W = super()._calc_weights(Xs)
        self.W_ = W

        n_samples = Ds[0].shape[0]
        n_dims = self.n_dims

        Ps = []
        for t in range(n_periods):
            P = _check_prepare_tsne(self, Xs[t])
            Ps.append(P)

        if not self.init is None:
            self.n_inits = 1

        best_cost = np.inf
        for i in range(self.n_inits):
            if self.verbose > 0:
                print("[{0}] Initialization {1}/{2}".format(self.method_str,i+1, self.n_inits))

            init = super()._initialize(Ps)
            for ith_try in range(self.max_tries):
                try:
                    # Set optimization arguments for early Exaggeration
                    opt_args = {
                        'init': init,
                        'method_str': self.method_str,
                        'n_iter': self.stop_lying_iter,
                        'n_iter_check': self.n_iter_check,
                        'eta': self.eta,
                        'momentum': self.initial_momentum,
                        'start_iter': 0,
                        'min_grad_norm': self.tol,
                        'verbose': self.verbose,
                        'kwargs': {
                            'static_cost_function': _kl_divergence, 
                            'Ds': [P * self.early_exaggeration for P in Ps],
                            'alpha': self.alpha, 
                            'p': self.p,
                            'weights' : W}}

                    Y, cost = gradient_descent_with_momentum(_evomap_cost_function, **opt_args)

                    opt_args.update({
                        'init': Y,
                        'n_iter': self.n_iter,
                        'momentum': self.final_momentum,
                        'start_iter': self.stop_lying_iter,
                        'kwargs': {
                            'static_cost_function': _kl_divergence, 
                            'Ds': Ps,
                            'alpha': self.alpha, 
                            'p': self.p,
                            'weights' : W}})

                    Y, cost = gradient_descent_with_momentum(_evomap_cost_function, **opt_args)              
                    break
                      
                except DivergingGradientError:
                    print("[{0}] Adjusting learning rate..".format(self.method_str))
                    self.eta /= 2
                    opt_args.update({'eta': self.eta})

                if ith_try == self.max_tries -1:
                    print("[{0}] ERROR: Gradient descent failed to converge.".format(self.method_str))
                    return -1

            if cost < best_cost:
                Ys = []
                for t in range(n_periods):
                    Ys.append(_get_positions_for_period(Y, n_samples, t))
                self.Ys_ = Ys
                self.cost_ = cost
                self.cost_static_ = super()._calc_static_cost(
                    Xs = Ps, Y_all_periods= Y, static_cost_function=_kl_divergence)
                self.cost_static_avg_ = self.cost_static_ / n_periods    
                best_cost = cost

        return self.Ys_