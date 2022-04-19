""" Python Implementation of Static T-SNE.

This module includes a raw-numpy implementation of t-SNE, adjusted for mapping of 
- high-dimensional data
- a distance matrix
- a similarity matrix

Main method inlcudes three representative examples. 

Todo:
	* Complete full documentation
	* Code refactoring

Author: Maximilian Peter Matthe. 
This version: May 2019.
"""

import numpy as np
import pylab
from scipy.spatial.distance import squareform

from numba import jit

EPSILON = 1e-12


def init_inclusions(Ds):
	# Create inclusion arrays if none provided
	inclusions = []
	for t in range(len(Ds)):
		inclusions.append(np.ones(Ds[0].shape[0]))
	return inclusions


def validate_input(Xs, inclusions):
	if len(Xs) != len(inclusions):
		raise ValueError("Unequal number of distance matrices and inclusions.")
		
	if not Xs[0].shape[0] == inclusions[0].shape[0]:
		raise ValueError("Inclusions do not match distance matrices.")

def avoid_identical_positions(Y):
    """ Slightly shuffle overlapping map positions. For some algorithms, e.g. SAMMON,
    identical map positions let the optimization fail (e.g., when gradients / hessians go to inf.)
    To avoid such problems, this method first looks for zero valued distances 
    on the map (except for self-distances) and slightly shuffles such objects
    by adding minimal random noise to them. 

    Args:
        Y (array): Map coordinates

    Returns:
        array: Map coordinates without identical positions
    """
    from scipy.spatial.distance import cdist
    import numpy as np    
    
    D_map = cdist(Y, Y)
    indices_i, indices_j = np.where(D_map <= 1e-8)
    shift_indices = []
    for i, idx_i in enumerate(indices_i):
        idx_j = indices_j[i]
        if idx_i != idx_j:
            shift_indices.append(idx_i) # only need to add one of both, as D is symmetric

    n_dims = Y.shape[1]
    n_shifts = len(shift_indices)
    if n_shifts > 0 :
        print("[EvoMap] Slightly shuffling {} objects to avoid zero valued map distances".format(n_shifts))

    Y[shift_indices, :] += np.random.normal(0,1e-2, (n_shifts, n_dims))

    return Y

    
def extend_gradient(inc_t, dY_static, n_dims):
	"""If some observations were excluded from the gradient computations, add
	0s for their gradients now."""
	
	n_samples = len(inc_t)
	if dY_static.shape[0] < n_samples:
		dY_static_full = np.zeros((n_samples, n_dims))
		k = 0
		for j in range(n_samples):
			if inc_t[j] != 0:
				dY_static_full[j, :] = dY_static[k, :]
				k += 1
	else:    
		dY_static_full = dY_static
		
	return dY_static_full

def initialize_positions(
	Xs, 
	n_dims = 2, 
	Y_inits = None, 
	inclusions = None, 
	verbose = 0):
	"""
	Parameters:
		Xs : list of length n_periods, each containing a distance matrix
		inclusions: list of length n_periods, each containg a 1D array of shape (n_samples) 
			with 0/1s indicating if the sample is included in this period
	"""
	n_periods = len(Xs)
	n_samples = Xs[0].shape[0]

	init_random = False
	if Y_inits is None:
		init_random = True

	Y_inits_array = np.zeros((n_samples, n_periods * n_dims))
	for t in range(n_periods):
		if init_random:
			# Try to initialize each period differently. If it does not work: Initialize them jointly
			Y_init_t = np.random.normal(0.0,1.0,size=(n_samples,n_dims))
		else: 
			Y_init_t = Y_inits[t]
			
		if not inclusions is None:
			Y_init_t[inclusions[t] == 0] = 0 # Initialize non-included objects at the origin
		
		Y_init_t = avoid_identical_positions(Y_init_t)
		Y_inits_array[:, (n_dims*t):(n_dims*t)+n_dims] = Y_init_t
			
	return Y_inits_array


def build_inclusion_array(inclusions, n_dims, Ys):
	"""
	Parameters:
		inclusions: list of length "n_periods", each containing a 1D array of shape (n_samples) 
		with 0/1s indicating if the sample should be included in this period 

	Returns:
		array (n_samples, n_periods)
	"""

	if inclusions is None:
		n_samples = Ys.shape[0]
		n_periods = Ys.shape[1] / n_dims
		n_periods = int(n_periods)
		
		inclusion_array = np.ones((n_samples, n_periods))
		
	else:
		n_periods = len(inclusions)
		n_samples = inclusions[0].shape[0]
		inclusion_array = np.ones((n_samples, n_periods))
		for t in range(n_periods):
			inclusion_array[:, t] = inclusions[t]
		
	return inclusion_array
    


@jit(nopython=True)
def Hbeta(D=np.array([]), beta=1.0):
	"""
		Compute the perplexity and the P-row for a specific value of the
		precision of a Gaussian distribution.
	"""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta)
	sumP = np.sum(P)
	H = np.log(sumP) + beta * np.sum(D * P) / sumP
	P = P / sumP
	return H, P

def calc_p_matrix(X, included, input_type, perplexity):
	""" Calculate joint-probability matrix from high-dimensional data / distances or similarities. 
	
	Arguments:
		X {float} -- (n,d) array of data or (n,n) array of pairwise similarities / distances
		input_type {str} -- one of ['data', 'similarity', 'distance']
		perplexity {int} -- target perplexity for determining sigmas 
	
	Returns:
		P [float] -- (n,n) array of joint-probabilites
	"""
	from . import _utils # Import here to avoid circular dependency when initializing the module

	n = X.shape[0]

	if input_type == "data":
		P = x2p(X, 1e-5, perplexity)
		#TODO: Add inclusion restrictions here

	elif input_type == "similarity":
		assert np.all(np.sum(X, axis = 0) != 0), "Zero Row(s) in Input Matrix"
		P = X

	elif input_type == "distance":
		assert np.all(np.sum(X, axis = 0) != 0), "Zero Row(s) in Input Matrix"
		D = X
		D = D[included == 1, :][:, included == 1]
		P = _utils._binary_search_perplexity(D.astype(np.float32), perplexity, 0)
		#P = d2p(D, 1e-5, perplexity)

		for i in range(n):
			if included[i] == 0:
				P = np.insert(P, i, 0*np.ones((1,P.shape[1])), 0)
				P = np.insert(P, i, 0*np.ones((1, P.shape[0])), 1)

	P = cond_to_joint(P)
	P = np.maximum(P, EPSILON)
	np.fill_diagonal(P,0)

	return P

@jit(nopython=True)
def calc_q_matrix(Y, inclusions):
	"""Calculate Q-Matrix of joint probabilities in low-dim space.
	
	Arguments:
		Y {np.ndarray} -- (n,2) array of map coordinates
		exclusions {np.ndarray} -- condensed-dist-mat indices for exclusions

	Returns:
		Q {np.ndarray} -- (n,n) array of joint probabilities in low-dim space.
		dist {np.ndarray} -- (n,n) array of squared euclidean distances
	"""
	n = Y.shape[0]
	dist = sqeuclidean_dist(Y)
#   dist = pdist(Y,"sqeuclidean")
	dist += 1
	dist **= -1
	
	# Mask non-included objects
	for idx in np.where(inclusions == 0)[0]:
		dist[idx, :] = 0
		dist[:, idx] = 0
		
	sum_dist = np.sum(dist)

	Q = np.maximum(dist/sum_dist, EPSILON)

	# Also return dist, to avoid recomputing it for the gradient
	return Q, dist

@jit(nopython=True)
def calc_gradient(Y, P, Q, dist):
	""" Calculate gradient of KL-divergence dC/dY.
	
	Arguments:
		Y {np.ndarray} -- (n,2) array of map coordinates
		P {np.ndarray} -- condensed-matrix of joint probabilities (high dim)
		Q {np.ndarray} -- condensed-matrix of joint probabilities (low dim)
		dist {np.ndarray} -- condensed-matrix of sq.euclidean distances
	
	Returns:
		dY {np.ndarray} -- (n,2) array of gradient values
	"""
	# Gradient: dC/dY
	(n_samples, n_dims) = Y.shape
	dY = np.zeros((n_samples, n_dims), dtype = np.float32)
	
	PQd = (P - Q) * dist
	for i in range(n_samples):
		dY[i] = np.dot(np.ravel(PQd[i]), Y[i] - Y)

	dY *= 4.0

	return dY

@jit(nopython=True)
def calc_sq_gradient(Y_t1, Y_t0, I_t1, I_t0):
	""" Calculate temporal component of gradient for Sequential t-SNE. 
	
	Arguments:
		Y_t1 {np.ndarray} -- map coordinates in t
		Y_t0 {np.ndarray} -- map coordinates in (t-1) 
		I_t1 {np.ndarray} -- inclusions in t (0/1 array)
		I_t0 {np.ndarray} -- inclusions in (t-1) (0/1 array)
	
	Returns:
		[np.array] -- Temporal component of gradient in t. Shape: (n,2)
	"""

	sq_gradient = np.zeros_like(Y_t1)

	if (Y_t0 is None) or (I_t0 is None):
		return sq_gradient
	else:
		sq_gradient = Y_t1 - Y_t0
		sq_gradient[I_t0 == 0] = 0
		sq_gradient[I_t1 == 0] = 0

	return sq_gradient

@jit(nopython=True)
def calc_dyn_cost(Ys, Ws, inclusions, p = 1, n_dims = 2):
	""" Calculate temporal component of the cost function 

	Returns:
		dyn_cost {float} -- value of the temporal cost function
	"""

	dyn_cost = 0
	n_samples = Ys.shape[0]
	n_periods = Ys.shape[1] / n_dims
	n_periods = int(n_periods)

	# Create a list of k-th order distances
	k_order_dist = []
	for k in range(p+1):

		# All First Order distances
		delta_Yt = np.zeros((n_periods,n_samples,n_dims))
		for t in range(k, n_periods):
			if k == 0:

				delta_Yt[t,:,:] = Ys[:, (n_dims*t):(n_dims*t)+n_dims] #0-order distance: Positions themselves

			else:
				delta_t = k_order_dist[k-1][t,:,:] - k_order_dist[k-1][t-1,:,:]
				delta_Yt[t,:,:] = delta_t #k-order distances

		k_order_dist.append(delta_Yt)
	
	# for each k, all k-th order distances are stored as a 3d array: (period, sample, dimension)
	# ALl entries for period < k are equal to zero (at least k periods are required to compute a kth order distance)

	for i in range(n_samples):
		for k in range(1, p+1):
			for t in range(k, n_periods):
				if inclusions[i, t] == 1:
					dyn_cost_it = Ws[i, 0] * np.linalg.norm(k_order_dist[k][t, i, :])
					dyn_cost += dyn_cost_it

	return dyn_cost

@jit(nopython=True)
def shift_elements(vector):
	"""Shift all elements in a vector by one (first element becomes zero). 
	"""
	last_element = 0
	new_vector = np.zeros_like(vector)
	for i in range(len(vector)):
		new_vector[i] = last_element
		last_element = vector[i]
	return new_vector
		
@jit(nopython=True)
def calc_dyn_gradient(Ys, Ws, inclusions, p = 1, n_dims = 2):
	""" Calculate temporal component of gradient. This part is independent of the mapping method of choice. 
	
	Parameters
	----------
	Ys: ndarray of shape (n_samples, n_dimensions * n_periods)
		Map positions. First n_dimensions columns correspond to first period, and so on.
	
	Ws : ndarray of shape (n_samples,n_dimensions*n_periods)
		Dynamic weights. First n_dimension columns correspond to first period, and so on. 
		
	"""

	dyn_grads = []
#   n_dims = Ys[0].shape[1]
#   n_samples = Ys[0].shape[0]
#   n_periods = len(Ys)
	n_periods = Ys.shape[1] / n_dims
	n_periods = int(n_periods)
	n_samples = Ys.shape[0]

	partial_delta_k = []
	for k in range(p+1):
		# Partial of delta k (for each k) w.r.t. t (for each t)
		partial_delta_k.append(np.zeros(n_periods))

	for k in range(1,p+1):
		if k == 1:
			partial_delta_k[k][0] = 1 # partial of 1st Order Distance w.r.t (t)
			partial_delta_k[k][1] = -1 # partial of 1st Order distance w.r.t. (t+1)
		else:
			partial_delta_k[k] = partial_delta_k[k-1] - shift_elements(partial_delta_k[k-1])

	# Create a list of k-th order distances
	# Each dictionary entry (for k in 1, ..., p)
	k_order_dist = []
	for k in range(p+1):

		# All First Order distances
		delta_Yt = np.zeros((n_periods,n_samples,n_dims))
		for t in range(k, n_periods):
			if k == 0:

				delta_Yt[t,:,:] = Ys[:, (n_dims*t):(n_dims*t)+n_dims] #0-order distance: Positions themselves

			else:
				delta_t = k_order_dist[k-1][t,:,:] - k_order_dist[k-1][t-1,:,:]
				delta_Yt[t,:,:] = delta_t #k-order distances

		k_order_dist.append(delta_Yt)

	for t in range(n_periods):
		dyn_grad = np.zeros((n_samples, n_dims))
		
		for k in range(1,p+1):  
			for tau in range(p+1):
				if (t+tau) < n_periods:
					dyn_grad += 2 * partial_delta_k[k][tau] * k_order_dist[k][t+tau, : ,: ]
					dyn_grad[inclusions[:,t+tau] == 0] = 0

#       dyn_grad = dyn_grad * Wsnp.repeat(Ws[t],n_dims).reshape((-1,n_dims))        
		dyn_grad = dyn_grad * Ws[:, (n_dims*t):(n_dims*t)+n_dims] # Pick columns 0 & 1 for period 0, columns 2 & 3 for period 1 
		dyn_grads.append(dyn_grad)

	return dyn_grads, k_order_dist



def calc_dyn_gradient_old(Ys, Ws, inclusions, p = 1, n_dims = 2):
	""" Calculate temporal component of gradient for Sequential t-SNE. 
	
	Arguments:
		Y_t1 {np.ndarray} -- map coordinates in t
		Y_t0 {np.ndarray} -- map coordinates in (t-1) 
		I_t1 {np.ndarray} -- inclusions in t (0/1 array)
		I_t0 {np.ndarray} -- inclusions in (t-1) (0/1 array)
	
	Returns:
		[np.array] -- Temporal component of gradient in t. Shape: (n,2)
	"""

	dyn_grads = []
	n_samples = Ys[0].shape[0]
	n_periods = len(Ys)

	partial_delta_k = []
	for k in range(p+1):
		# Partial of delta k (for each k) w.r.t. t (for each t)
		partial_delta_k.append(np.zeros(n_periods))

	for k in range(1,p+1):
		if k == 1:
			partial_delta_k[k][0] = 1 # partial of 1st Order Distance w.r.t (t)
			partial_delta_k[k][1] = -1 # partial of 1st Order distance w.r.t. (t+1)
		else:
			partial_delta_k[k] = partial_delta_k[k-1] - shift_elements(partial_delta_k[k-1])

	# Create a list of k-th order distances
	# Each dictionary entry (for k in 1, ..., p)
	k_order_dist = []
	for k in range(p+1):

		# All First Order distances
		delta_Yt = np.zeros((n_periods,n_samples,n_dims))
		for t in range(k, n_periods):
			if k == 0:

				delta_Yt[t,:,:] = Ys[t] #0-order distance: Positions themselves

			else:
				delta_t = k_order_dist[k-1][t,:,:] - k_order_dist[k-1][t-1,:,:]
				delta_Yt[t,:,:] = delta_t #k-order distances

		k_order_dist.append(delta_Yt)

	for t in range(n_periods):
		dyn_grad = np.zeros_like(Ys[0])
		
		for k in range(1,p+1):	
			for tau in range(p+1):
				if (t+tau) < n_periods:
					dyn_grad += 2 * partial_delta_k[k][tau] * k_order_dist[k][t+tau, : ,: ]
					dyn_grad[inclusions[t+tau] == 0] = 0
				
		dyn_grad = dyn_grad * np.repeat(Ws[t],n_dims).reshape((-1,n_dims))
		dyn_grads.append(dyn_grad)

	return dyn_grads, k_order_dist

def kl_divergence(P, Q, included):
	"""" Calculate KL-Divergence between Q and P.
	
	Arguments:
		Q {np.ndarray} -- condensed joint-probability matrix (low-dim) 
		P {np.ndarray} -- condensed  joint-probability matrix (high-dim)
		exclusions {np.array} -- array of condensed-dist-mat indices for exclusions
	
	Returns:
		kl_divergence {float} -- kl-divergence
	"""
	# Calculate exclusion indices for condensed matrix forms:

	n = P.shape[0]
	mask = np.eye(n, dtype = 'bool')
	indices = np.where(included==0)[0]
	for idx in indices:
		mask[idx,:] = True
		mask[:, idx] = True
	
	p = P[~mask]
	q = Q[~mask]
	kl_divergence = np.sum(p*np.log(p/q))
	return kl_divergence

def d2p(D, tol = 1e-5, perplexity = 30.0):
	"""
		Performs a binary search to get P-values in such a way that each
		conditional Gaussian has the same perplexity.
	"""

	# Initialize some variables
	n = D.shape[0]
	P = np.zeros((n, n))
	beta = np.ones((n, 1))
	logU = np.log(perplexity)

	# Loop over all datapoints
	for i in range(n):

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf
		betamax = np.inf
		Di = D[i, :] # all distances, except to itself
		Di = np.delete(Di, i)
		(H, thisP) = Hbeta(Di, beta[i])

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU
#		assert np.isfinite(Hdiff), "FAILED AT " + str(i)
		tries = 0
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = np.copy(beta[i])[0]
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2.
				else:
					beta[i] = (beta[i] + betamax) / 2.
			else:
				betamax = np.copy(beta[i])[0]
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2.
				else:
					beta[i] = (beta[i] + betamin) / 2.

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i])
			Hdiff = H - logU
			tries += 1

		# Set the final row of P-values1
		thisP = np.insert(thisP, i, 0) 
		P[i, :] = thisP

	return P

@jit(nopython=True)
def sqeuclidean_dist(Y):
	n = Y.shape[0]
	d = Y.shape[1]

	D = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			for k in range(d):
				D[i,j] += (Y[i,k] - Y[j,k])**2

	return D

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
	"""
		Performs a binary search to get P-values in such a way that each
		conditional Gaussian has the same perplexity.
	"""

	print("[t-SNE] Computing pairwise distances...")
	(n, d) = X.shape
	sum_X = np.sum(np.square(X), 1)
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

	return d2p(D, tol, perplexity)


def pca(X=np.array([]), no_dims=50):
	"""
		Runs PCA on the NxD array X in order to reduce its dimensionality to
		no_dims dimensions.
	"""

	print("[t-SNE] Preprocessing the data using PCA...")
	(n, d) = X.shape
	X = X - np.tile(np.mean(X, 0), (n, 1))
	(l, M) = np.linalg.eig(np.dot(X.T, X))
	Y = np.dot(X, M[:, 0:no_dims])
	return Y

@jit(nopython=True)
def cond_to_joint(P):
	""" Take an asymmetric conditional probability matrix and convert it to a 
	symmetric joint probability matrix.

	Symmetrizes and normalizes the matrix. 
	""" 
	np.fill_diagonal(P,0)                  # Set diagonal to zero
	P = P + P.T
	sum_P = np.maximum(np.sum(P), EPSILON)
	P = np.maximum(P / sum_P, EPSILON)
	return P 


def is_valid_input(X, input_type):
	""" Check input for validity.

	Checks distance / similarity matrix for symmetry and non-negativity.
	
	Arguments:
		X {float} -- Input data for tsne
		input_type {str} -- 'data', 'similaritiy' or 'distance'
	
	Returns:
		valid [bool] -- Indicating if input is valid
	"""
	valid = True
	(m,n) = X.shape

	if not X.dtype == 'float64':
		raise ValueError("Wrong input data type. Expected float, got " + X.dtype)

	if input_type == 'distance':
		if np.any(X < 0):
			raise ValueError("All distances should be non-negative.")

		if m != n:
			raise ValueError("Non-symmetric distance matrix")

	if input_type == 'similarity':
		if np.any(X < 0):
			raise ValueError("All similarities should be non-negative.")
		if m != n:
			raise ValueError("Non-symmetric similarity matrix.")
	
	return valid

def calc_weights_new(X_ts, n_dims, weighting_scheme):
	n_samples = X_ts[0].shape[0]
	n_periods = len(X_ts)
	W = np.zeros((n_samples))
	if weighting_scheme is None:
		W = np.ones((n_samples))
	else:

		for t in range(1, n_periods):
			delta = X_ts[t] - X_ts[t-1]
			delta = np.power(delta, 2)
			delta = delta.sum(axis = 1)
			delta = delta.reshape(n_samples)
			W += delta
			
		if weighting_scheme == 'inverse':
			W = np.power(W, -1)
		elif weighting_scheme == 'inverse_plus':
			W = np.power(W+1, -1)
		elif weighting_scheme == 'mirror':
			W = np.max(W) - W
		elif weighting_scheme == 'exponential':
			lamb = 1/np.max(W) # If all goes wrong, readjust to lamb = 5 / np.max(W)
			W = np.exp(-(lamb * W))
		else:
			raise ValueError("Unkown weighting scheme: {}.".format(weighting_scheme))

	# Normalize to [0,1]:
	W = W / np.max(W)
	Ws = np.repeat(W, n_dims * n_periods).reshape(n_samples, n_dims * n_periods)
	return Ws

def calc_weights(X_ts, n_dims, weighting_scheme):
	""" Calculate temporal penalty weight for t>=1, based on the hellinger distance of each row of the P_i|j matrix. 
	
	Arguments:
		X_ts {list} -- list of (n,n)-arrays, representing the p_i|j matrices
	
	Returns:
		Ws {list}-- list of (n,1)-arrays, with the temporal weights.
	"""
	
	EPSILON = np.float_power(10,-8) # Eps. is used to avoid division by zero in inverse weighiting
	NEIGHBORHOOD_SIZE = 10
	n_samples = X_ts[0].shape[0]
	n_periods = len(X_ts)
	Ws = np.ones((n_samples, n_dims * n_periods))

	for t in range(n_periods):
		
		X_this = X_ts[t]
		X_prev = X_ts[t-1]
		if (not X_this.shape[0] == X_this.shape[1]):
			X_this = squareform(X_this)
			X_prev = squareform(X_prev)
	
		n = X_this.shape[0]
		
		if t == 0:
			W_t = np.ones((n,1))
		
		else:
			if weighting_scheme == "relations":
				# Calculate squared euclidean distance between matric rows
				W_t = X_this - X_prev
				W_t = np.power(W_t,2)
				W_t = np.sum(W_t, axis = 1)
				W_t = np.sqrt(W_t)
				W_t = W_t.reshape(n,1)
				
				# Inverse weighting
				W_t = W_t + EPSILON
				W_t = np.power(W_t,-1)
				W_t = W_t / np.sum(W_t) * n

			elif weighting_scheme == "neighbors":

				W_t = np.zeros((n,1))
				for i in range(n):
					# get NN in previous period
					# Argsort sorts indices from low to high --> The rightmost indices have the highest similarity
					# shift selection of top N indices by 1 to the left, since the last index is always the element itself 
					nn_previous = np.argsort(X_prev[i,:])[-(NEIGHBORHOOD_SIZE+1):-1] 
					nn_this = np.argsort(X_this[i,:])[-(NEIGHBORHOOD_SIZE+1):-1]
					
					intersec = [id for id in nn_previous if id in nn_this]
					union = np.union1d(nn_previous,nn_this)
					W_t[i] = len(intersec)/len(union) 
		
			elif weighting_scheme == "nearest_relations":

				W_t = np.zeros((n,1))
				for i in range(n):
					# get NN in previous period
					# Argsort sorts indices from low to high --> The rightmost indices have the highest similarity
					# shift selection of top N indices by 1 to the left, since the last index is always the element itself 
					nn_previous = np.argsort(X_prev[i,:])[-(NEIGHBORHOOD_SIZE+1):-1] 
					nn_this = np.argsort(X_this[i,:])[-(NEIGHBORHOOD_SIZE+1):-1]
					
					union = np.union1d(nn_previous,nn_this)
					delta = X_this[i, union] - X_prev[i, union]
					delta = np.power(delta, 2)
					delta = np.sum(delta)
					delta = np.sqrt(delta)
					delta = delta + EPSILON
					delta = np.power(delta, -1)
					W_t[i] = delta 

				W_t = W_t / np.sum(W_t) * n

			elif weighting_scheme == "kl_divergence":

				W_t = np.zeros((n,1))
				for i in range(n):
					# get NN in previous period
					# Argsort sorts indices from low to high --> The rightmost indices have the highest similarity
					# shift selection of top N indices by 1 to the left, since the last index is always the element itself 
					p = X_this[i, :]
					q = X_prev[i, :]
					p = np.maximum(p, EPSILON)
					q = np.maximum(q, EPSILON)
					p = p/p.sum()
					q = q/q.sum()

					delta = np.sum(p*np.log(p/q))
					delta = delta + EPSILON
					delta = np.power(delta, -1)
					W_t[i] = delta 

				W_t = W_t / np.sum(W_t) * n

			elif weighting_scheme == "nearest_kl_divergence":

				W_t = np.zeros((n,1))
				for i in range(n):
					# get NN in previous period
					# Argsort sorts indices from low to high --> The rightmost indices have the highest similarity
					# shift selection of top N indices by 1 to the left, since the last index is always the element itself 
					nn_previous = np.argsort(X_prev[i,:])[-(NEIGHBORHOOD_SIZE+1):-1] 
					nn_this = np.argsort(X_this[i,:])[-(NEIGHBORHOOD_SIZE+1):-1]
					
					union = np.union1d(nn_previous,nn_this)
					p = X_this[i, union]
					q = X_prev[i, union]
					p = np.maximum(p, EPSILON)
					q = np.maximum(q, EPSILON)
					p = p/p.sum()
					q = q/q.sum()
					delta = np.sum(p*np.log(p/q))
					delta = delta + EPSILON
					delta = np.power(delta, -1)
					W_t[i] = delta 

				W_t = W_t / np.sum(W_t) * n
		
		# Broadcast weights in t on each dim in t
		Ws[:, (n_dims*t):(n_dims*t)+n_dims] = W_t

	return Ws
