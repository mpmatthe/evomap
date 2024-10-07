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
    
    @staticmethod
    def _cmdscale(D, n_dims, eps=1e-16):
        """                                                                                       
        Classical multidimensional scaling (MDS)                                                  
                                                                                                
        Parameters                                                                                
        ----------                                                                                
        D : (n, n) array                                                                          
            Symmetric distance matrix.                                                            
                                                                                                
        Returns                                                                                   
        -------                                                                                   
        Y : (n, p) array                                                                          
            Configuration matrix. Each column represents a dimension. Only the                    
            p dimensions corresponding to positive eigenvalues of B are returned.                 
            Note that each dimension is only determined up to an overall sign,                    
            corresponding to a reflection.                                                        
                                                                                                
        e : (n,) array                                                                            
            Eigenvalues of B.                                                                     
                                                                                                
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
        self.Y_, _ = self._cmdscale(X, self.n_dims)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.Y_
