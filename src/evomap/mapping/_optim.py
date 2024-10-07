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
    """Gradient descent optimization with backtracking line search via halving.

    This function performs gradient descent optimization to minimize the objective 
    function, using a backtracking line search to adaptively adjust the 
    step size. 

    Parameters
    ----------
    objective : callable
        The objective function to be minimized. It should return both the cost 
        (value) and the gradient when called. The function signature is expected 
        to be `objective(Y, *args, **kwargs)` where `Y` is the current map 
        coordinates and `args` and `kwargs` are additional arguments.
    init : ndarray of shape (n_samples, n_dims)
        The initial starting point for the optimization (e.g., initial coordinates).
    n_iter : int
        The total number of gradient descent iterations to perform.
    n_iter_check : int, optional
        The frequency at which the progress of the optimization is reported, 
        by default 50.
    max_halves : int, optional
        The maximum number of times to halve the step size during backtracking, 
        by default 10.
    step_size : float, optional
        The initial step size for the gradient descent updates, by default 1.
    min_grad_norm : float, optional
        The tolerance level for stopping the optimization based on the gradient 
        norm, by default 1e-7.
    verbose : int, optional
        The verbosity level of the function's output:
        - 0: No output
        - 1: Only final status messages
        - 2: Detailed iteration-by-iteration progress, by default 0.
    method_str : str, optional
        A string to identify the method, useful for logging and output messages, 
        by default "" (empty string).
    args : list, optional
        Additional positional arguments passed to the objective function, 
        by default None.
    kwargs : dict, optional
        Additional keyword arguments passed to the objective function, 
        by default None.

    Returns
    -------
    ndarray of shape (n_samples, n_dims)
        The optimized map coordinates (final positions in the reduced space).
    float
        The final value of the cost function after optimization.

    Raises
    ------
    DivergingGradientError
        If the gradient norm becomes excessively large, indicating divergence.
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
    """Gradient descent optimization with momentum.

    This function performs gradient descent with momentum to optimize an objective 
    function. 

    Parameters
    ----------
    objective : callable
        The objective function to be minimized. It should return both the cost 
        (value) and the gradient when called. The function signature is expected 
        to be `objective(Y, *args, **kwargs)` where `Y` is the current map 
        coordinates and `args` and `kwargs` are additional arguments.
    init : ndarray of shape (n_samples, n_dims)
        The initial starting point for the optimization (e.g., initial coordinates).
    n_iter : int
        The total number of gradient descent iterations to perform.
    start_iter : int, optional
        The starting iteration, useful for resuming optimization from a certain 
        point, by default 0.
    n_iter_check : int, optional
        The frequency at which the progress of the optimization is reported, 
        by default 50.
    momentum : float, optional
        The momentum factor that determines how much of the previous step's 
        velocity is retained. Higher values increase momentum, by default 0.8.
    eta : float, optional
        The learning rate, or step size, that controls how much the parameters 
        are adjusted in each iteration, by default 50.
    min_grad_norm : float, optional
        The tolerance level for stopping the optimization based on the gradient 
        norm, by default 1e-7.
    verbose : int, optional
        The verbosity level of the function's output:
        - 0: No output
        - 1: Only final status messages
        - 2: Detailed iteration-by-iteration progress, by default 0.
    method_str : str, optional
        A string to identify the method, useful for logging and output messages, 
        by default "" (empty string).
    args : list, optional
        Additional positional arguments passed to the objective function, 
        by default None.
    kwargs : dict, optional
        Additional keyword arguments passed to the objective function, 
        by default None.

    Returns
    -------
    ndarray of shape (n_samples, n_dims)
        The optimized map coordinates (final positions in the reduced space).
    float
        The final value of the cost function after optimization.

    Raises
    ------
    DivergingGradientError
        If the gradient norm becomes excessively large, indicating divergence.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    # Initialize Optimization Variables
    Y =  init.copy()
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    if verbose > 1:
        print("[{0}] Gradient descent with Momentum: {1}".format(method_str, momentum))

    # Run Gradient Descent
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
    """Print the progress of the optimization during iterative updates.

    Parameters
    ----------
    iter : int
        The current iteration of the optimization process.
    method_str : str
        A string identifier for the method being used, useful for logging 
        or distinguishing between different optimization methods.
    cost : float
        The current value of the cost function being minimized.
    grad_norm : float, optional
        The norm of the gradient at the current iteration, by default None. 
        If provided, it is displayed in the progress report.
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

