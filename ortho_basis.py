import numpy as np
import scipy.integrate as integrate
from itertools import combinations_with_replacement

class OrthoBasis(object):
    """A class for orthonormalizing an arbitrary set of one-dimensional 
    functions within an arbitrary interval using the Gram-Schmidt process 
    and numerical integration. The defined scalar product is
    \int_a^b v1(x) v2(x) dx. The method could be overwritten in order
    change the scalar product definition or to adjust numerical parameters
    as the process is numerically unstable. The stability should be always
    checked!

    Parameters
    ----------
    interval : tuple(, iterable)
        Integration interval [a, b].

    modified_gs : bool, default True
        If True, the 'modified' Gram-Schmidt implementaion is used which 
        is numeracially more stable, however, computationally more demanding 
        than the straightforward implementaion.

    Attributes
    -----------
    b_list : list of functions, [n_basis]
        List of orthonormalized functions.

    T : array, [n_basis, n_basis]

    Methods
    -------
    fit(v_list) : v_list : list
                      List of functions to be orthonormalized.
        Performs the Gram-Schmidt process. 

    transform(X) : X, array, [n_data, n_basis]
        Transforms matrix X (outputs of input functions v_list for n_data
        data points in [a, b]) to outputs of orthonormalized basis set via 
        linear mapping. Returns the transformed matrix.

    get_scalar_product(v1, v2) : v1, v2 : functions
        Returns scalar product between v1 and v2: \int_a^b v1(x) v2(x) dx.

    get_corr_matrix()
        Returns a symmetric matrix of scalar products between the 
        orthonormalized functions.

    """
    def __init__(self, interval=(0, 1), modified_gs=True):
        self.interval = interval
        self.modified_gs = modified_gs
    
    def get_scalar_product(self, v1, v2):
        return integrate.quad(lambda x: v1(x) * v2(x), *self.interval,)[0]
    
    def fit(self, v_list):
        self.b_list = []
        self.n_basis = len(v_list)
        
        # transformation matrix
        self.T = np.eye(self.n_basis)

        for i_basis in range(self.n_basis):
            if self.modified_gs:
                coefs = self._modified_implementation(v_list, i_basis)
            else:
                coefs = self._straightforward_implementation(v_list, i_basis)

            # new orthogonal basis vector
            b = self._get_linear_combi(v_list[: i_basis + 1], coefs)
            
            # normalize
            norm = np.sqrt(self.get_scalar_product(b, b))
            self.T[i_basis, : i_basis + 1] = coefs / norm
            b = self._div(b, norm)

            self.b_list.append(b)

    def transform(self, X):
        return np.dot(X, self.T.T)
    
    def get_corr_matrix(self):
        C = np.zeros((self.n_basis, self.n_basis))
        for i, j in combinations_with_replacement(range(self.n_basis), 2):
            C[i, j] = self.get_scalar_product(self.b_list[i], self.b_list[j])
            if i != j:
                C[j, i] = C[i, j]
        return C
 
    def _get_linear_combi(self, v_list, c_list):
        return lambda x: sum([c_list[i] * v(x) for i, v in enumerate(v_list)])

    def _div(self, v, val):
        return lambda x: v(x) / val
 
    def _straightforward_implementation(self, v_list, i_basis):
        # get scalar products of i_basis'th input vector onto new basis vectors from self.b_list
        scalar_products = np.array([self.get_scalar_product(v_list[i_basis], self.b_list[j]) for j in range(i_basis)])
        scalar_products = np.append(-scalar_products, 1.)

        # combine scalar products with coeffiecients of orthononormal basis vectors
        # from previos iteration steps to new coefficients (of current step) for linear 
        # tranformation of input basis to new basis.
        coefs = np.dot(self.T[: i_basis + 1, : i_basis + 1].T, scalar_products)
        
        return coefs

    def _modified_implementation(self, v_list, i_basis):
        coefs = self.T[i_basis, : i_basis + 1].copy()

        # build new orthogonal basis vector iteratively
        for j, b_prev in enumerate(self.b_list):
            # get scalar products of current status of new basis vector onto orthogonal basis vectors from self.b_list
            b = self._get_linear_combi(v_list[: i_basis + 1], coefs)
            scalar_product = self.get_scalar_product(b, b_prev)

            # combine scalar product with coeffiecient of orthononormal basis vectors
            # from previos iteration steps to new coefficients (of current i_basis step) for linear 
            # tranformation of input basis to new basis.
            coefs[: j+1] -= scalar_product * self.T[j, : j+1]
        return coefs

   
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    v_list = []
    exponents = np.arange(4)
    for d in exponents: 
        def f(x, d=d):
            return x**d
        v_list.append(f)
     #   print(np.sqrt(ob.get_scalar_product(f, f)))
    #exit()
    ob = OrthoBasis(interval=(-1, 1), modified_gs=True) 
    ob.fit(v_list)

    colors = ['g', 'r', 'b', 'k']
    x = np.linspace(-1, 1, 100)
    
    # get outputs in orthonormal basis set B either through linear mapping
    # of outputs of reference basis set
    F = x[..., np.newaxis]**exponents
    B = ob.transform(F)

    # or calculate input directly in orthonormal basis set
    B = np.transpose([b(x) for b in ob.b_list])

    # Analytical solution for orthonormal basis set (for comparison)
    B_analytical = [np.ones(x.size) * 1/np.sqrt(2),
                    np.sqrt(3. / 2.)                   * x,
                    np.sqrt(45. / 8.)                  * (x**2 - 1. / 3.),
                    np.sqrt(1. / (2. / 7. - 6. / 25.)) * (x**3 - 3. / 5. * x)]
    
    for i in exponents:
        plt.plot(x, B_analytical[i], '%s-'  %colors[i])
        plt.plot(x, B[:, i], '%s:' %colors[i], linewidth=3)

    plt.show()
