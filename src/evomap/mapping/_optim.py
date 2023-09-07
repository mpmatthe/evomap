"""
Core functions shared within mapping module. Mostly related to different
optimization routines implemented and adjusted for mapping. 
"""

import numpy as np
EPSILON = 1e-12

class Error(Exception):
    """Base class for other exceptions"""
    pass

class DivergingGradientError(Error):
    """Raised when the input value is too small"""
    pass

def gradient_descent_line_search(
    objective,
    init,
    n_iter,
    n_iter_check = 50,
    max_halves = 10,
    step_size = 1,
    min_grad_norm = 1e-7,
    verbose = 0,
    method_str = "",
    args = None,
    kwargs = None):
    """ Gradient descent with backtracking via halving. 

    Optimizes the objective function iteratively. At each step, a halving
    procedure is used to ensure that step sizes are set such that cost values
    decrease. 

    Parameters
    ----------
    objective : callable
        Function to be optimized. Expected to return the function value and the
        gradient when called. See examples for exact syntax. 
    init : ndarray of shape (n_samples, n_dims)
        Starting initialization.
    n_iter : int
        Total number of gradient descent iterations.
    n_iter_check : int, optional
        Interval in which cost values are reported, by default 1
    max_halves : int, optional
        Maximum number of halving steps in line search, by default 10
    step_size : int, optional
        Initial step size, by default 1
    min_grad_norm : float, optional
        Error tolerance, by default 1e-7
    verbose : int, optional
        Level of verbosity, by default 0
    method_str : str, optional
        Method label, by default ""
    args : list, optional
        Arguments passed to the objective function, by default None
    kwargs : dict, optional
        Keyword arguments passed to the objective function, by default None

    Returns
    -------
    ndarray of shape (n_samples, n_dims)
        Final map coordinates

    float
        Final cost function value
    """
    import numpy as np
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    Y = init.copy()
    if verbose > 0:
        print("[{0}] Running Gradient Descent with Backtracking via Halving".format(method_str))

    # Run Gradient Descent
    init_step_size = step_size
    kwargs["compute_error"] = True
    kwargs["compute_grad"] = False

    error_old, _ = objective(Y, *args, **kwargs)
    for i in range(0, n_iter):
        report_progress = (i + 1) % n_iter_check == 0
        kwargs["compute_error"] = True
        kwargs["compute_grad"] = True

        error, grad = objective(Y, *args, **kwargs)

        if np.any(grad >= 1/EPSILON):
            print('[{0}] Diverging gradient norm at iteration {1}'.format(method_str, i+1))
            raise DivergingGradientError()
        Y_old = Y
        step_size = init_step_size
        for j in range(max_halves):
            Y = Y_old - (step_size * grad)
            
            kwargs["compute_grad"] = False
            error, _ = objective(Y, *args, **kwargs)
            if error < error_old:
                break
            else:
                step_size = step_size * 0.5

        error_old = error
        grad_norm = np.linalg.norm(grad) * step_size
        if verbose > 1:
            if report_progress:
                report_optim_progress(
                    iter = i, 
                    method_str = method_str, 
                    cost = error, 
                    grad_norm = grad_norm)

        if grad_norm <= min_grad_norm:
            if verbose > 0:
                print("[{0}] Iteration {1}: gradient norm vanished. Final cost: {2:.2f}".format(method_str, i+1, error))
            break
        elif i==n_iter-1:
            if verbose > 0:
                print("[{0}] Maximum number of iterations reached. Final cost: {1:.2f}".format(method_str, error))

    return Y, error

def gradient_descent_with_momentum(objective,
    init,
    n_iter,
    start_iter = 0,
    n_iter_check = 50,
    momentum = .8,
    eta = 50,
    min_grad_norm = 1e-7,
    verbose = 0,
    method_str = "",
    args = None,
    kwargs = None):
    """ Gradient descent with momentum.

    Optimize the objective function using momentum-based gradient descent, 
    as used, for instance, in t-SNE.

    Parameters
    ----------
    objective : callable
        Function to be optimized. Expected to return the function value and the
        gradient when called. See examples for exact syntax. 
    init : ndarray of shape (n_samples, n_dims)
        _description_
    n_iter : int
        Total number of gradient descent iterations.
    start_iter : int, optional
        Startint iteration, if optimization (re-)starts at a later stage
        , by default 0
    n_iter_check : int, optional
        Interval in which cost values are reported, by default 50
    momentum : float, optional
        Momentum factor, by default .8
    eta : int, optional
        Learning rate, by default 50
    min_grad_norm : float, optional
        Error tolerance, by default 1e-7
    verbose : int, optional
        Level of verbosity, by default 0
    method_str : str, optional
        Method label, by default ""
    args : list, optional
        Arguments passed to the objective function, by default None
    kwargs : dict, optional
        Keyword arguments passed to the objective function, by default None

    Returns
    -------
    ndarray of shape (n_samples, n_dims)
        Final map coordinates

    float
        Final cost function value
    """

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    #------------- Initialize Optimization Variables ---------------------#

    Y =  init.copy()
    iY = np.zeros_like(Y)        # Step sizes
    gains = np.ones_like(Y)

    if verbose > 1:
        print("[{0}] Gradient descent with Momentum: {1}".format(method_str, momentum))

    #------------- Run Gradient Descent ---------------------#
    for iter in range(start_iter, n_iter):

        report_progress = (iter + 1) % n_iter_check == 0
        kwargs["compute_error"] = True
        kwargs["compute_grad"] = True

        error, grad = objective(Y, *args, **kwargs)

        if np.any(grad >= 1/EPSILON):
            print('[{0}] Diverging gradient norm at iteration {1}'.format(method_str, iter+1))
            raise DivergingGradientError()

        dec = np.sign(iY) == np.sign(grad)
        inc = np.invert(dec)

        gains[inc] += .2
        gains[dec] *= .8

        iY = momentum * iY - eta * (gains * grad)
        Y = Y + iY 

        grad_norm = np.linalg.norm(grad)
        if report_progress:
            if verbose > 1:
                report_optim_progress(
                    iter = iter, 
                    method_str = method_str, 
                    cost = error, 
                    grad_norm = grad_norm)

        if grad_norm <= min_grad_norm:
            if verbose > 0:
                print("[{0}] Iteration {1}: gradient norm vanished.".format(method_str, iter+1, grad_norm))
                break
        elif iter==n_iter-1:
            if verbose > 0:
                print("[{0}] Maximum number of iterations reached. Final cost: {1:.2f}".format(method_str, error))

    return Y, error



def report_optim_progress(iter, method_str, cost, grad_norm = None):
    """Print optimization progress. 

    Parameters
    ----------
    iter : int
        Current iteration.
    method_str : str
        Method label. 
    cost : float
        Current cost function value
    grad_norm : float, optional
        Gradient norm, by default None
    """
    if cost < 1e3:
        outstr = "[{0}] Iteration {1} -- Cost: {2:.2f}".format(
            method_str, iter+1, cost)
    else:
        outstr = "[{0}] Iteration {1} -- Cost: {2:.2e}".format(
            method_str, iter+1, cost)


    if not grad_norm is None:
        if grad_norm < 1e3:
            outstr += " -- Gradient Norm: {0:.4f}".format(grad_norm)
        else:
            outstr += " -- Gradient Norm: {0:.4e}".format(grad_norm)

    print(outstr)

