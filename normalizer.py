import numpy as np

def get_normalized(X, *args):
    norm = Normalizer()
    norm.fit(X)
    if not args:
        return norm.transform(X)
    else:
        return [norm.transform(x) for x in (X,) + args]

class Standardizer(object):
    """Class for standardizing matrix to mean 0 and variance 1."""
    def __init__(self, with_mean=True, with_std=True, tol=0.0001):
        self.scales = 1.
        self.means = 0.
        self.with_mean = with_mean
        self.with_std = with_std
        self.tol = tol
    
    def fit(self, X):
        """ Calculate means and scales of X and returns standardized X.

            Parameters
            ----------
            X : 2darray
                Matrix to be standardized.

            Returns
            -------
            Xstan : 2darray
                    Standardized Matrix.
        """
        if self.with_mean:
            self.means  = X.mean(axis=0)
        if self.with_std:
            self.scales = X.std(axis=0)
            self.scales[abs(self.scales) < self.tol] = 1.
    
    def fit_n_get(self, X): 
        """ Calculate means and scales of X and returns standardized X.
            Assuming that X is needed standardized, fit_n_get(X) is faster
            than fit(X) and transform(X).
        """
        if self.with_mean:
            self.means = X.mean(axis=0)
            X_stan = (X - self.means)
    
            if self.with_std:
                self.scales = np.linalg.norm(X_stan, axis=0)/np.sqrt(len(X_stan))
                self.scales[abs(self.scales) < self.tol] = 1.
                X_stan = X_stan / self.scales
            return X_stan
        
        elif self.with_std:
            self.scales = X.std(axis=0)
            return X / self.scales
        
        else:
            return X
    
    def transform(self, X): 
        return (X - self.means) / self.scales
    
    def transform_parameters(self, coefs, bias): 
        coefs_trans = coefs * self.scales[:, np.newaxis]
        bias_trans  = bias + np.dot(self.means, coefs) 
        return coefs_trans, bias_trans

    def invert(self, X):
        """Transform X back"""
        return X * self.scales + self.means
        
    def invert_parameters(self, coefs, bias):
        """ transform coefs of linear model with standardized input to coefs of non-standardized input"""
        #TODO check if it is really np.dot in bias_inv
        coefs_inv = coefs / self.scales[:, np.newaxis] 
        bias_inv  = bias - np.dot(self.means, coefs_inv)
        return coefs_inv, bias_inv

class Normalizer(object):
    """Class for standardizing matrix to mean 0 and variance 1."""
    def __init__(self, tol=0.0001):
        self.scales = 1.
        self.tol = tol
    
    def fit(self, X):
        """ Calculate means and scales of X and returns standardized X.

            Parameters
            ----------
            X : 2darray
                Matrix to be standardized.
        """
        self.scales = np.linalg.norm(X, axis=0)
        self.scales[abs(self.scales) < self.tol] = 1.
    
    def transform(self, X): 
        return X / self.scales
    
    def transform_parameters(self, coefs, non_negative=None): 
        if non_negative is None:
            return coefs * self.scales
        else:
            return coefs * np.sqrt(self.scales)
    
    def invert_parameters(self, coefs, non_negative=None):
        """ transform coefs of linear model with normalized input to coefs of non-normalized input"""
        if non_negative is None:
            return coefs / self.scales
        else:
            return coefs / np.sqrt(self.scales)

