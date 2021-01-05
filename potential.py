import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from descriptor import Descriptor
from scipy.interpolate import RegularGridInterpolator
from regressor import Regressor
import ase
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.energy_models import EwaldElectrostaticModel

class Potential(Descriptor, Calculator):
    """Module that fits 2b+3b potential based on Descriptor module.
    it, furthermore, can be used as an ase calculator.
    Note, that energy unit is per cell.

    Target/Input unit is per atom.


    Parameters
    ----------
    all Descriptor parameters, see Descritpor module

    finite_diff_forces: float
        Finite difference for calculating numerical forces.

    finite_diff_stress: float
        Finite difference for calculating numerical stress.

    Methods
    -------
    All ase calculator methods (get_potential_energy, get_forces), 
    if atoms calculator is set with Potential module.

    All Desriptor module methods.

    fit(atoms_list, Y, G=None, **kwargs): list of ase atoms objects,
                                          Y: target energy (per cell),
                                          G: precalculated structural 
                                             descriptor matrix
                                          kwargs: Regressor arguments
    
    predict(atoms_list=None, G=None): list of ase atoms objects,
                                      G: precalculated structural
                                         descriptor matrix
    predict_train(): 
        get predited values for training set.

    get_G(atoms_list): list of ase atoms objects
        Get structural descriptor matrix.
    
    get_g(atoms): ase atoms object
        Get structural descriptor.

    map_potential(lower_bound=None, n=10): 
                lower_bound: float, lower bound of x, y and z of 3b 
                             descriptor for interpolation.
                n: int, number of interpolation grid points per axis
        Interpolates 3b potential for (strong) speed up.
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, finite_diff_forces=0.01, finite_diff_stress=0.01, atoms_init=None, madelung=None, 
                intercept=0.,
                **kwargs):

        Descriptor.__init__(self, **kwargs)
        Calculator.__init__(self)

        self.finite_diff_forces = finite_diff_forces
        self.finite_diff_stress = finite_diff_stress

        if atoms_init is not None:
            self.init_atomic_numbers(atoms_init)
        
        self.is_avg_energy = True
        self.mapped = False

        self.intercept = intercept
        self.madelung = madelung

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        self.results['energy'] = self._get_energy(atoms)
        self.results['free_energy'] = self.results['energy'] * 1

        if 'forces' in properties:
            self.results['forces'] = self._get_forces(atoms)
            
        if 'stress' in properties:
            self.results['stress'] = self._get_stress(atoms)
    
    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def _get_energy(self, atoms):
        if self.mapped:
            energy = self._get_mapped_energy(atoms)
        else:
            energy = np.dot(self.get_g(atoms), self.regressor.coefs)

        if self.madelung is not None:
            energy += self._get_emad(atoms)

        energy += self.intercept

        if not self.is_avg_energy:
            energy *= len(atoms)

        return energy
 
    def _get_forces(self, atoms):
        F = self.calculate_numerical_forces(atoms, d=self.finite_diff_forces)
        if self.is_avg_energy:
            F *= len(atoms)
        return F
    
    def _get_stress(self, atoms):
        F = self.calculate_numerical_stress(atoms, d=self.finite_diff_stress, voigt=False)
        if self.is_avg_energy:
            F *= len(atoms)
        return F
    

    def _get_emad(self, atoms):

        c = sorted(set(atoms.get_chemical_symbols()))
        at_py = AseAtomsAdaptor.get_structure(atoms)
        at_py.add_oxidation_state_by_element(self.madelung)
        ewald = EwaldElectrostaticModel()
        e = ewald.get_energy(at_py) / len(atoms)
        return e


    def fit(self, atoms_list, Y, G=None):
        if isinstance(atoms_list[0], ase.atoms.Atoms):
            self.G_train = self.get_G(atoms_list)
        else:
            self.G_train = atoms_list

        if self.intercept is 'center':
            self.intercept = Y.mean()
            
        self.regressor.fit(self.G_train, Y - self.intercept)
    
    def init_regressor(self, **kwargs):
        self.regressor = Regressor(**kwargs)

    def get_gs(self, atoms):
        structural_descripor = self.get_structural_descriptor(atoms)
        return [structural_descripor[comb] for comb in self.atomic_numbers]

    def get_g(self, atoms):
        return np.concatenate(self.get_gs(atoms))
    
    def get_G(self, atoms_list):
        G = []
        for atoms in atoms_list:
            G.append(self.get_g(atoms))
        return np.array(G)

    def predict_train(self):
        pred = self.regressor.predict(self.G_train)
        pred *= self.N_at_train
        return pred + self.intercept
        
    def predict(self, atoms_list):
        if isinstance(atoms_list[0], ase.atoms.Atoms):
            G = self.get_G(atoms_list)
        else:
            G = atoms_list
 

        return self.regressor.predict(G) + self.intercept

    def map_potential(self, lower_bound=None, n=10):
        """Interpolates 3b potentials for speed up. 
        Linear interpolation is used as it was found much faster
        than tested cubic spline codes for having similar accuracy.
        """
        if lower_bound is None:
            try:
                lower_bound = self.ortho_2b[0]
            except:
                lower_bound = 1.
        else:
            try:
                lower_bound /= self.fac
            except:
                pass

        if self.cut_to_sym:
            n_b = n
            r_cut_b = self.r_cut_3b
        else:
            n_b = 2 * n
            r_cut_b = 2 * self.r_cut_3b

        a = np.linspace(lower_bound, self.r_cut_3b, n  )
        b = np.linspace(lower_bound,       r_cut_b, n_b)
        X, Y, Z = np.meshgrid(a, a, b, indexing='ij')
        d = np.transpose([X.flatten(), Y.flatten(), Z.flatten()])[:, np.newaxis,:]
        dd = {'desc': d}
        
        n_3b_terms = len(self.atomic_numbers) - self.i_split
        G = self.sum_environmental_to_structural(dd, n_body=3, return_nosum=True, not_julia=True)
        coefs_split = np.split(self.regressor.coefs[-G.shape[1]*n_3b_terms:], n_3b_terms)
        energies = [np.dot(G, coefs).reshape(n, n, n_b) for coefs in coefs_split]
        itp = [RegularGridInterpolator((a, a, b), e, method='linear') for e in energies]
        self.maps = dict(zip(self.atomic_numbers[self.i_split:], itp))
        self.mapped = True

    def _get_mapped_energy(self, atoms):
        desc = self.get_environmental_descriptor(atoms)

        g_2b = np.concatenate([self.sum_environmental_to_structural(desc[comb], n_body=2)
                                     for comb in self.atomic_numbers[:self.i_split]])
        e_2b = np.dot(g_2b, self.regressor.coefs[: g_2b.size])
        e_3b = sum([self.maps[comb](desc[comb]['desc'].squeeze(axis=1)).sum()
                                     for comb in self.atomic_numbers[self.i_split:]])
        return (e_2b + e_3b) / len(atoms)

    def split_2b3b(self, atoms):
        atoms_list = [atoms]
        if isinstance(atoms_list[0], ase.atoms.Atoms):
            G = self.get_G(atoms_list)
        else:
            G = atoms_list

        return self.regressor.get_contrib_split(G)








