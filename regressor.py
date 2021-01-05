import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from normalizer import Normalizer, get_normalized

class Regressor(object):
    """Specific regressor that can perform ridge cv with intercept False 
    but normalize True (in contrast to RidgeCV from sklearn). Furtermore,
    also a hyper parameter optimizer based on either cross validation 
    or a fixed validaton set is included. The two hyperparameters
    optimized on a square grid are the regularization parameter lambda 
    of ridgre regression and a weight on columns, e.g. to specify how
    strong the 3b contribution should be in comparison to 2b.
    """
    def __init__(self, lamda=None, lambdas=None, cv=10, normalize=True, weight_columns=None, split=None, indices_train_vali=None, non_negative=None, coefs=None):

        self.normalize = normalize

        self.lamda  = lamda
        self.lambdas = lambdas
        self.cv = cv

        self.weight_columns = weight_columns
        self.split = split
        self.indices_train_vali = indices_train_vali

        self.non_negative = non_negative

        self.coefs = coefs

    def save(self):
        np.savetxt('Coefs.dat', self.coefs)
        np.savetxt('LamWeight.dat', [self.lam, self.weight_columns])
    
    def load(self, folder='./'):
        self.coefs = np.loadtxt(folder+'Coefs.dat')
        self.lam, self.weight_columns = np.loadtxt(folder+'LamWeight.dat')
    
    def fit(self, X, Y):
        if self.lambdas is None:
            if self.non_negative is None:
                self.coefs = np.linalg.lstsq(X, Y, rcond=-1)[0]
            else:   
                self.coefs = nnls(X * self.non_negative, Y)[0]
        else:
            if not isinstance(self.weight_columns, list):
                if isinstance(self.lambdas, list):
                    self.lam = self.ridge_cv(X, Y)
                else:
                    self.lam = self.lambdas
                    
            else:
                self.lam, self.weight_columns = self.ridge_cv_weights(X, Y)

            if self.normalize:
                normalizer = Normalizer()
                normalizer.fit(X)
                X_transformed = normalizer.transform(X)
            else:
                X_transformed = X *1
            if self.weight_columns is not None:
                X_transformed[:, self.split:] *= self.weight_columns
    
            ridge = Ridge(alpha=self.lam, fit_intercept=False, normalize=False)
            ridge.fit(X_transformed, Y)
    
            # transform coefs back
            if self.normalize:
                self.coefs = normalizer.invert_parameters(ridge.coef_)
            else:
                self.coefs = ridge.coef_

            if self.weight_columns is not None:
                self.coefs[self.split:] *= self.weight_columns   

    def predict(self, X):
        if self.non_negative is None:
            return np.dot(X, self.coefs) 
        else:
            return np.dot(X * self.non_negative, self.coefs) 

    def ridge_cv(self, X, Y): 
        errors_matrix = np.empty((self.cv, len(self.lambdas) ))
        kf = KFold(n_splits=self.cv, shuffle=True)
    
        for i_cv, (train_index, test_index) in enumerate(kf.split(Y)):
            if self.normalize:
                X_train, X_test = get_normalized(X[train_index], X[test_index])
                if self.weight_columns is not None:
                    X_train[:, self.split:] *= self.weight_columns
                    X_test[:, self.split:]  *= self.weight_columns
            else:
                X_train, X_test = X[train_index], X[test_index]

            for i_lam, lam in enumerate(self.lambdas):
                ridge = Ridge(alpha=lam, fit_intercept=False, normalize=False)
                ridge.fit(X_train, Y[train_index])
                pred = ridge.predict(X_test)
    
                errors_matrix[i_cv, i_lam] = get_squared_error(pred, Y[test_index])
        #self, in case it is needed from outside
        self.lambda_errors = errors_matrix.sum(0)
        optimal_lambda = self.lambdas[self.lambda_errors.argmin()]
    
        return optimal_lambda

    def ridge_cv_weights(self, X, Y): 
        if self.indices_train_vali is None:
            errors_matrix = np.empty((self.cv, len(self.lambdas), len(self.weight_columns)))
            kf = KFold(n_splits=self.cv, shuffle=True)
            indices_train_vali = kf.split(Y)
        else:
            errors_matrix = np.empty((len(self.indices_train_vali), len(self.lambdas), len(self.weight_columns)))
            
            indices_train_vali = self.indices_train_vali
    
        for i_cv, (train_index, test_index) in enumerate(indices_train_vali):
            for i_weight_columns, weight_columns in enumerate(self.weight_columns):
                if self.normalize:
                    X_train, X_test = get_normalized(X[train_index], X[test_index])
                    X_train[:, self.split:] *= weight_columns
                    X_test[:,  self.split:] *= weight_columns
        
                else:
                    X_train, X_test = X[train_index], X[test_index]

                for i_lam, lam in enumerate(self.lambdas):
                    ridge = Ridge(alpha=lam, fit_intercept=False, normalize=False)
                    ridge.fit(X_train, Y[train_index])
                    pred = ridge.predict(X_test)
        
                    errors_matrix[i_cv, i_lam, i_weight_columns] = get_squared_error(pred, Y[test_index])
        
        self.lambda_errors = errors_matrix.sum(0)
        i_min_lam, i_min_weights = np.unravel_index(np.argmin(self.lambda_errors, axis=None), self.lambda_errors.shape)

        optimal_lambda = self.lambdas[i_min_lam]
        weight_columns = self.weight_columns[i_min_weights]
    
        return optimal_lambda, weight_columns

    def get_contrib_split(self, X):
        pred_1 = np.dot(X[:, :self.split], self.coefs[:self.split])
        pred_2 = np.dot(X[:, self.split:], self.coefs[self.split:])
        return np.transpose([pred_1, pred_2])

def get_squared_error(a, b, axis=None):
    """ Get sum of squared deviations (errors) between two
        arrays.
        
        Parameters
        ----------
        a, b : ndarray

        axis : int, optional, default None
               Determines along which axis the SD
               should be computed.

        Returns
        -------
        rmse : float or ndarray
               SD(s) between a and b.
        """
    return np.linalg.norm(a - b, axis=axis)**2

