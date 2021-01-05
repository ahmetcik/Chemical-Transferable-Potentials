from ase.neighborlist import neighbor_list
import numpy as np
from itertools import product

def get_random_structure(atoms_in, max_strain=0.2, d_min=1.6):
    """Extracted from published scripts on http://www.libatoms.org/Home/DataRepository 
    for modeling Silicon, i.e. A. P. Bartok et. al, Phys.Rev. X 8, 041048 (2018)."""
    atoms = atoms_in.copy()
    
    F = 2.0 * max_strain * (np.random.rand(3,3) - 0.5)
    F_diag = F.trace() / 3.
    F[0,0] += 1.0 - F_diag
    F[1,1] += 1.0 - F_diag
    F[2,2] += 1.0 - F_diag
    F = 0.5*(F+F.T)
    atoms.set_cell(np.dot(atoms.get_cell(),F))
        
    atoms.set_scaled_positions(np.random.rand(len(atoms), 3)) 
    while neighbor_list('d', atoms, d_min).size > 0: # while unphysically close atoms
        atoms.set_scaled_positions(np.random.rand(len(atoms), 3)) 

    return atoms


def get_3b_centers(rmin, rmax, steps=10, sigma=1.):
    a = np.linspace(rmin, rmax, steps)
    b = np.linspace(rmin, rmax, steps)
    c = np.linspace(rmin, rmax*2, steps*2)
    D = np.array([[x, y, z] for x, y, z in product(a, b, c)])
    sigmas = np.ones(len(D)) * sigma
    return D, sigmas

