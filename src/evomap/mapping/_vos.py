import numpy as np
from scipy.spatial.distance import cdist
from numba import jit

class VOS():

    def __init__(
        self,
        n_dims = 2,
        n_iter = 1000,
        n_iter_check = 1,
        step_size = 1,
        lmb = 0,
        min_grad_norm = 1e-7,
        verbose = 0):
        
        self.n_dims = n_dims
        self.n_iter = n_iter
        self.lmb = lmb
        self.n_iter_check = n_iter_check
        self.step_size = step_size
        self.min_grad_norm = min_grad_norm
        self.verbose = verbose

    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, S):
        
        n_samples = S.shape[0]
        n_dims = self.n_dims
        init = np.random.normal(0,.1,(n_samples, n_dims))
        #init = init * 1 / _dist_sum(init)

        # Set arguments passed to objective function in grad descent
        opt_args = {
            'init': init,
            'n_iter': self.n_iter,
            'n_iter_check': self.n_iter_check,
            'step_size': self.step_size,
            'min_grad_norm': self.min_grad_norm,
            'verbose': self.verbose,
            'args': [S, self.lmb]
        }

        Y, cost = _gradient_descent(_cost_function, **opt_args)

        self.Y_ = Y
        self.cost_ = cost

        return self.Y_

@jit(nopython=True)
def _constraint(Y, D): 
   
    constraint = np.sum(np.triu(D, k = -1))
    constraint = constraint - 1
    constraint = constraint
    return constraint
  
@jit(nopython=True)
def _constraint_grad(Y, D):

    grad_constraint = np.zeros_like(Y)
    n_samples = Y.shape[0]
    for k in range(n_samples):
        grad_k = np.zeros_like(grad_constraint[k,:])
        for i in range(k-1):
            Y_dis =  Y[i,:] - Y[k,:]
            grad_k -= Y_dis / np.linalg.norm(Y_dis)
            
        for j in range(k+1,n_samples):
            Y_dis = Y[k,:] - Y[k, j]
            grad_k += Y[k,:] / np.linalg.norm(Y[k,:])

        grad_constraint[k,:] = grad_k

    return grad_constraint

def _cost_function(Y, S, lmb = 0, compute_error = True):
    D = cdist(Y,Y, metric = 'euclidean')
    D_sq = cdist(Y,Y, metric = 'sqeuclidean')
    if compute_error:
        SD = S*D_sq
        cost = np.sum(np.triu(SD, k = -1))
        cost -= lmb * _constraint(Y, D)
    else:
        cost = None

    grad = np.zeros_like(Y)
    n_samples = Y.shape[0]
    for k in range(n_samples):
        Y_k = Y[k,:] 
        dis = Y_k - Y
        grad[k,:] = np.dot(S[k,:], 2*dis)  

    grad -= lmb * _constraint_grad(Y, D) 

    return cost, grad

def _dist_sum(Y):
    dis = cdist(Y,Y, metric = 'euclidean')
    constraint = np.sum(np.triu(dis, k = -1))
    return constraint

def _gradient_descent(
    objective,
    init,
    n_iter,
    n_iter_check = 1,
    step_size = 1,
    min_grad_norm = 1e-7,
    verbose=0,
    args = None,
    kwargs = None
):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    Y = init.copy()

    for i in range(0, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == n_iter - 1

        error, grad = objective(Y, *args, **kwargs)
        grad_norm = np.linalg.norm(grad)

        # Gradient update
        Y = Y - grad * step_size
        # Projection step
       # Y = Y * 1 / _dist_sum(Y)

        if check_convergence:
            print(
                "[VOS] Iteration %d: error = %.7f,"
                " gradient norm = %.7f"
                " (%s iterations)"
                % (i + 1, error, grad_norm, n_iter_check)
            )

            if grad_norm <= min_grad_norm:
                print(
                    "[VOS] Iteration %d: gradient norm %f. Finished."
                    % (i + 1, grad_norm)
                )
                break

    return Y, error