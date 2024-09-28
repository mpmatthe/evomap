import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _binary_search_perplexity_numba(
    sqdistances : np.ndarray, 
    desired_perplexity : float, 
    n_steps : int = 100, 
    perplexity_tolerance : float = 1e-5, 
    epsilon_dbl : float = 1e-8
):
    """
    Binary search for sigmas of conditional Gaussians.
    This approximation reduces the computational complexity from O(N^2) to O(uN).

    Parameters
    ----------
    sqdistances : np.ndarray, shape (n_samples, n_neighbors), dtype=np.float32
        Distances between training samples and their k nearest neighbors.
        When using the exact method, this is a square (n_samples, n_samples)
        distance matrix. The TSNE default metric is "euclidean" which is
        interpreted as squared euclidean distance.
    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.
    n_steps : int, optional
        Maximum number of binary search steps. Default is 100.
    perplexity_tolerance : float, optional
        Tolerance for the difference between computed and desired entropy. Default is 1e-5.
    epsilon_dbl : float, optional
        Small constant to prevent division by zero. Default is 1e-8.

    Returns
    -------
    P : np.ndarray, shape (n_samples, n_neighbors), dtype=np.float32
        Probabilities of conditional Gaussian distributions p_i|j.
    beta_sum : float
        Sum of beta values for all samples, used to compute mean sigma.
    """
    # Parameters
    n_samples = sqdistances.shape[0]
    n_neighbors = sqdistances.shape[1]
    using_neighbors = n_neighbors < n_samples

    # Initialize variables
    beta_sum = 0.0
    desired_entropy = np.log(desired_perplexity)

    # Initialize P as float64 for higher precision during calculations
    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)

    # Iterate over each sample
    for i in prange(n_samples):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Binary search to find the optimal beta
        for l in range(n_steps):
            # Compute current entropy and corresponding probabilities
            # computed just over the nearest neighbors or over all data
            # if we're not using neighbors
            sum_Pi = 0.0

            # Compute probabilities
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = np.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            # Avoid division by zero
            if sum_Pi == 0.0:
                sum_Pi = epsilon_dbl

            sum_disti_Pi = 0.0

            # Normalize probabilities and compute weighted sum
            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            # Compute entropy and difference from desired entropy
            entropy = np.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            # Check if the entropy difference is within tolerance
            if np.abs(entropy_diff) <= perplexity_tolerance:
                break

            # Adjust beta based on entropy difference
            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        # Accumulate beta for mean sigma calculation
        beta_sum += beta

    # Cast P to float32 for consistency with original Cython function
    P = P.astype(np.float32)

    return P, beta_sum

def _binary_search_perplexity(sqdistances, desired_perplexity, verbose=0, n_steps=100, perplexity_tolerance=1e-5, epsilon_dbl=1e-8):
    """
    Wrapper function to handle verbosity and compute mean sigma.

    Parameters
    ----------
    sqdistances : np.ndarray, shape (n_samples, n_neighbors), dtype=np.float32
        Distances between training samples and their k nearest neighbors.
        When using the exact method, this is a square (n_samples, n_samples)
        distance matrix. The TSNE default metric is "euclidean" which is
        interpreted as squared euclidean distance.
    desired_perplexity : float
        Desired perplexity (2^entropy) of the conditional Gaussians.
    verbose : int
        Verbosity level.
    n_steps : int, optional
        Maximum number of binary search steps. Default is 100.
    perplexity_tolerance : float, optional
        Tolerance for the difference between computed and desired entropy. Default is 1e-5.
    epsilon_dbl : float, optional
        Small constant to prevent division by zero. Default is 1e-8.

    Returns
    -------
    P : np.ndarray, shape (n_samples, n_neighbors), dtype=np.float32
        Probabilities of conditional Gaussian distributions p_i|j.
    mean_sigma : float
        Mean sigma value across all samples.
    """

    # follow the Numba function signature to prevent slow re-compilation for every call
    P, beta_sum = _binary_search_perplexity_numba(
        np.ascontiguousarray(sqdistances, dtype=float),
        float(desired_perplexity),
        int(n_steps),
        float(perplexity_tolerance),
        float(epsilon_dbl)
    )
    
    n_samples = sqdistances.shape[0]
    mean_sigma = np.sqrt(n_samples / beta_sum)
    
    if verbose:
        print(f"[t-SNE] Mean sigma: {mean_sigma:.6f}")
    
    return P

# compile numba function on fake data call
#_ = _binary_search_perplexity(np.ones((100,100)), 1)
