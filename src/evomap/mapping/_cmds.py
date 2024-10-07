"""
Classic (SVD-based) Multidimensional Scaling, as proposed in:

Torgerson, W.S. Multidimensional scaling: I. Theory and method. Psychometrika 17, 401–419 (1952).

Thanks to Francis Song, from whom this implementation has borrowed. Source: http://www.nervouscomputer.com/hfs/cmdscale-in-python/
"""

from __future__ import division 
import numpy as np

class CMDS():

    def __init__(self, n_dims = 2):
        self.n_dims = 2

    def __str__(self):
        """Create a string representation of the CMDS instance."""
        result = f"CMDS(n_dims={self.n_dims})"
        return result

    @staticmethod
    def _cmdscale(D, n_dims, eps=1e-16):
        """Perform classical multidimensional scaling (CMDS) on the input distance matrix.

        CMDS reduces the dimensionality of a distance matrix while
        preserving the pairwise distances as well as possible using eigenvalue decomposition.

        Parameters
        ----------
        D : np.array of shape (n, n)
            Symmetric distance matrix to be scaled.
        n_dims : int
            Number of dimensions to which the data should be reduced.
        eps : float, optional, default=1e-16
            Tolerance for numerical precision in rounding the resulting coordinates.

        Returns
        -------
        Y : np.array of shape (n, n_dims)
            Configuration matrix with the reduced dimensionality representation of the points.
        e : np.array of shape (n,)
            The eigenvalues corresponding to the dimensions.

        Raises
        ------
        ValueError
            If the input matrix `D` is not square or symmetric.
        """
        # Number of points                                                                        
        n = len(D)
    
        # Centering matrix                                                                        
        H = np.eye(n) - np.ones((n, n))/n
    
        # YY^T                                                                                    
        B = -H.dot(D**2).dot(H)/2
    
        # Diagonalize                                                                             
        evals, evecs = np.linalg.eigh(B)
    
        # Sort by eigenvalue in descending order                                                  
        idx   = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]

        # Enforce consistent sign for each eigenvector
        for i in range(n_dims):
            # Find the index of the component with the largest absolute value
            max_idx = np.argmax(np.abs(evecs[:, i]))
            # If the component is negative, flip the eigenvector
            if evecs[max_idx, i] < 0:
                evecs[:, i] *= -1

        # Select positive eigenvalues and corresponding eigenvectors
        positive_evals = evals[:n_dims]
        positive_evecs = evecs[:, :n_dims]
    
        # Compute the coordinates using positive-eigenvalued components only
        L = np.diag(np.sqrt(positive_evals))
        Y = positive_evecs.dot(L)
    
        return np.round(Y, -int(np.log10(eps))), positive_evals

    def fit(self, X):
        """Fit the CMDS model to the provided distance matrix.

        Parameters
        ----------
        X : np.array of shape (n, n)
            Symmetric distance matrix to be scaled.

        Returns
        -------
        self : object
            Returns the instance itself with the configuration matrix `Y_` stored as an attribute.
        """
        self.Y_, _ = self._cmdscale(X, self.n_dims)
        return self

    def fit_transform(self, X):
        """Fit the CMDS model to the distance matrix and return the transformed coordinates.

        Parameters
        ----------
        X : np.array of shape (n, n)
            Symmetric distance matrix to be scaled.

        Returns
        -------
        np.array of shape (n, n_dims)
            The transformed coordinates (configuration matrix) in the reduced dimensional space.
        """
        self.fit(X)
        return self.Y_
