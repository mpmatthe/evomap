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
    def _cmdscale(D, n_dims):
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
    
        # Compute the coordinates using positive-eigenvalued components only                      
        w, = np.where(evals > 0)
        L  = np.diag(np.sqrt(evals[w]))
        V  = evecs[:,w]
        Y  = V.dot(L)
    
        return Y[:, :n_dims], evals[evals > 0]

    def fit(self, X):
        self.Y_, _ = self._cmdscale(X, self.n_dims)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.Y_
