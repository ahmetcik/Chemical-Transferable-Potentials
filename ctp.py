import numpy as np
from potential import Potential
from ctnn import ChemicalTransferNeuralNetwork
from ortho_basis import OrthoBasis
from normalizer import Normalizer, Standardizer
import re
from ase.data import atomic_numbers
import ase

class CTP(object):
    """ 
    Target unit is per atoms.

    In case pyjulia is installed, calculating descriptor is one order of magnitude faster 
    if python-jl is run.
    """
    def __init__(self, r_cut_2b=5., r_cut_3b=3.,                         # Descriptor parameters
                 cutoff_width_2b=1., cutoff_width_3b=1.,                 # Descrptor parameters
                 r_centers=None, sigmas=None, degrees=None, ortho_2b=None,  # Descrptor parameters
                 degrees_eam=None, sigma_eam=3., ortho_eam=None,
                 lmax=1, nmax=3, shift_3d=True, cut_to_sym=False,        # Descrptor parameters
                 r_centers_3b=None, sigmas_3b=None,
                 ortho_3b=None,
                 ortho_basis_2b=None, ortho_basis_eam=None,# Descrptor parameters
                 ortho_basis_3b=None,
                 finite_diff_forces=0.01, finite_diff_stress=0.01,       # Potential parameters
                 pot_weights=[1., 0.1, 0.001],
                 activation='relu', std_init_weights=0.1,                # Neural network parameters
                 non_negative=None,                                      # Neural network parameters
                 architecture_2b=(100,), architecture_3b=(100,),         # Neural network parameters
                 architecture_eam=(100,),         # Neural network parameters
                 atomic_info=None,
                 fac=None):                                     # Neural network parameters

        # Although potential parameters could have been passed via **kwargs
        # it is good to see all paremters when looking up the CTP module.
        self.kwargs_pot = dict(r_cut_2b=r_cut_2b, r_cut_3b=r_cut_3b,
                               cutoff_width_2b=cutoff_width_2b, cutoff_width_3b=cutoff_width_3b,
                               r_centers=r_centers, sigmas=sigmas, degrees=degrees, ortho_2b=ortho_2b,
                               degrees_eam=degrees_eam, sigma_eam=sigma_eam, ortho_eam=ortho_eam,
                               lmax=lmax, nmax=nmax, shift_3d=shift_3d, cut_to_sym=cut_to_sym,
                               ortho_3b=ortho_3b, ortho_basis_3b=ortho_basis_3b,
                               ortho_basis_2b=ortho_basis_2b, ortho_basis_eam=ortho_basis_eam, fac=fac,
                               r_centers_3b=r_centers_3b, sigmas_3b=sigmas_3b)
        self.potentials = {}
        
        # get ortho basis from some helper potential and use for all potentials
        self.pot_info = Potential(**self.kwargs_pot)
       # self.kwargs_pot['ortho_basis'] = self.pot_info.ortho_basis
        self.pot_weights = pot_weights
        # rearrange atomic info matrix suct that atomic number corresponds to row index
        # of matrix
        self.C = self._get_atomic_info(atomic_info)
        
        architecture_2b, architecture_3b, architecture_eam = self._get_architecture(architecture_2b, 
                                                                                    architecture_3b, 
                                                                                    architecture_eam)
        self.nn = ChemicalTransferNeuralNetwork(activation=activation, 
                                                architecture_2b=architecture_2b,
                                                architecture_3b=architecture_3b, 
                                                architecture_eam=architecture_eam, 
                                                pot_weights=pot_weights,
                                                std_init_weights=std_init_weights, 
                                                non_negative=non_negative)
   
    def _get_architecture(self, architecture_2b, architecture_3b, architecture_eam):
        n_basis_2b  = self.pot_info.get_n_basis_2b()
        n_basis_3b  = self.pot_info.get_n_basis_3b()
        n_basis_eam = self.pot_info.get_n_basis_eam()
        n_feat_2b   = self.C.shape[1] *2
        n_feat_3b   = self.C.shape[1] *2 # in the feature maybe different
        n_feat_eam  = self.C.shape[1] *2 # in the feature maybe different
        architecture_2b  = (n_feat_2b, *architecture_2b, n_basis_2b)
        architecture_3b  = (n_feat_3b, *architecture_3b, n_basis_3b)
        architecture_eam = (n_feat_eam, *architecture_eam, n_basis_eam)
        return architecture_2b, architecture_3b, architecture_eam
    
    def _get_atomic_info(self, atomic_info):
        Z = atomic_info[:, 0].astype(int)
        features = atomic_info[:, 1:]
        atomic_info_new = np.zeros((Z.max()+1, features.shape[1]))
        atomic_info_new[Z] = features
        self.Z = Z
        return atomic_info_new

    def set_nn_train_para(self, optimizer_type='adam', learning_rate=0.001, 
                          n_epochs=200000, stopping_threshold=0., batch_size=None, 
                          lambda_reg=0., keep_2b=1., keep_3b=1., epoch_save_weights=None, 
                          decay=None, epoch_start_3b=None):
        
        self.nn.init_nn(optimizer_type=optimizer_type, learning_rate=learning_rate, 
                        n_epochs=n_epochs, stopping_threshold=stopping_threshold, 
                        batch_size=batch_size, lambda_reg=lambda_reg, 
                        keep_2b=keep_2b, keep_3b=keep_3b, epoch_save_weights=epoch_save_weights, 
                        decay=decay, epoch_start_3b=epoch_start_3b)
    
    def get_atomic_numbers_single(self, atoms, make_binary=True):
        atnu = sorted(set(atoms.numbers))
        # put elemental into framework of binaries
        if make_binary and len(atnu) == 1:
            atnu *= 2
        return tuple(atnu)
    
    def get_atomic_numbers(self, atoms_list, make_binary=True):
        return [self.get_atomic_numbers_single(atoms, make_binary=make_binary) 
                for atoms in atoms_list]

    def add_potentials(self, atoms_list):
        compounds = self.get_atomic_numbers(atoms_list)
        compounds_new = set(compounds) - set(self.potentials.keys())
        new_potentials = {compound: Potential(atoms_init=compound, **self.kwargs_pot)
                                for compound in compounds_new}
        self.potentials.update(new_potentials) 
    
    def get_g(self, atnu, atoms):
        gs = self.potentials[atnu].get_gs(atoms)
        if len(set(atnu)) == 1:
            gs = [gs[0]/3. for _ in range(3)] + [gs[1]/6. for _ in range(6)]
        return np.concatenate(gs)

    def get_structural_input(self, atoms_list):
        compounds = self.get_atomic_numbers(atoms_list)
        G = [self.get_g(compound, at) for compound, at in zip(compounds, atoms_list)]
        return np.array(G)
        
    def get_X(self, atoms_list):
        compounds = self.get_atomic_numbers(atoms_list)
        self.add_potentials(atoms_list)
        G = self.get_structural_input(atoms_list)
        return np.hstack((compounds, G))

    def split_G(self, G):
        i_split = [self.pot_info.get_n_basis_2b(), self.pot_info.get_n_basis_eam()] * 3 \
                + [self.pot_info.get_n_basis_3b()] * 5
        i_split = np.cumsum(i_split)
        i_swap = [0, 2, 4, 1, 3, 5] + list(range(6, 12))

        Gs = np.split(G, i_split, axis=1)
        Gs = [Gs[i] for i in i_swap]

        return Gs
    
    def get_nn_input(self, X):
        if isinstance(X[0], ase.atoms.Atoms):
            X = self.get_X(X)
        
        compounds = X[:, :2]
        G = X[:, 2:]
        Gs = self.split_G(G)
        
        Xs = self.get_chemical_input(compounds.astype(int))
        return Xs, Gs

    def check_atomic_numbers(self, compounds):
        diff = set(compounds.flatten()) - set(self.Z)
        if len(diff) != 0:
            raise ValueError("The following atomic numbers are missing in the atomic_info matrix: %s" % diff)
    
    def get_chemical_input(self, compounds):
        self.check_atomic_numbers(compounds)

        combinations  = [[0, 0], [0, 1], [1, 1]] #2b
        combinations += [[i, *comb] for i in range(2) 
                           for comb in combinations] #3b

        Xs = []
        for i, comb in enumerate(combinations):
            if len(comb) == 2:
                z1, z2 = compounds.T[comb]
                X1 =     self.C[z1] + self.C[z2]
                X2 = abs(self.C[z1] - self.C[z2])
            else:
                z1, z2, z3 = compounds.T[comb]
                X1 = 2*self.C[z1] + self.C[z2] + self.C[z3]
                X2 = abs(self.C[z1] - self.C[z2]) + abs(self.C[z1] - self.C[z3])
                
            Xs.append( np.hstack((X1, X2)) )

            if i == 2:
                Xs.extend(Xs)

        return Xs
 

    def get_chemical_input_atom(self, compounds):
        self.check_atomic_numbers(compounds)

        combinations  = [[0, 0], [0, 1], [1, 1]] #2b
        combinations += [[i, *comb] for i in range(2) 
                           for comb in combinations] #3b

        Xs = []
        for i, comb in enumerate(combinations):
            if len(comb) == 2:
                z1, z2 = compounds.T[comb]
                X1 =     self.C[z1] + self.C[z2]
                X2 = abs(self.C[z1] - self.C[z2])
            else:
                z1, z2, z3 = compounds.T[comb]
                X1 = 2*self.C[z1] + self.C[z2] + self.C[z3]
                X2 = abs(self.C[z1] - self.C[z2]) + abs(self.C[z1] - self.C[z3])
                
            Xs.append( np.hstack((X1, X2)) )

            if i == 2:
                z1, z2 = compounds.T[[0, 1]]
                c1 = self.C[z1]
                c2 = self.C[z2]

                Xs.extend([c1, c1*0, c2])

        return Xs
 

    def fit(self, X, Y, vali=None, XG=None):
        if XG is None:
            Xs, Gs = self.get_nn_input(X)
            
            if vali is not None:
                X_vali, Y_vali = vali
                Xs_vali, Gs_vali = self.get_nn_input(X_vali)
                vali = [Xs_vali, Gs_vali, Y_vali] 
        else:
            Xs, Gs = XG
            if vali is not None:
                XG_vali, Y_vali = vali
                Xs_vali, Gs_vali = XG_vali
                vali = [Xs_vali, Gs_vali, Y_vali] 
           

            
        self.nn.fit(Xs, Gs, Y, vali=vali)

    def predict(self, X, XG=None):
        if XG is None:
            Xs, Gs = self.get_nn_input(X)
        else:
            Xs, Gs = XG
        return self.nn.predict(Xs, Gs)
    
    def predict_2b(self, X, XG=None):
        if XG is None:
            Xs, Gs = self.get_nn_input(X)
        else:
            Xs, Gs = XG


        return self.nn.predict_2b(Xs, Gs)

    def predict_2beam(self, X,XG=None):
        if XG is None:
            Xs, Gs = self.get_nn_input(X)
        else:
            Xs, Gs = XG

        return self.nn.predict_2beam(Xs, Gs)

    def get_compound(self, obj):
        if isinstance(obj, str):
            compound = [atomic_numbers[s] for s in re.findall('[A-Z][^A-Z]*', obj)]
        elif isinstance(obj, ase.atoms.Atoms):
            compound = obj.numbers
        elif isinstance(obj, (tuple, list, np.ndarray)):
            compound = obj

        compound = tuple(sorted(compound))
        if len(compound) == 1:
            compound *= 2
        return compound 

    def get_coefs(self, compound):
        if len(compound) == 1:
            compound *= 2
        compound = np.array([compound])
        Xs = self.get_chemical_input(compound)
        coefs = self.nn.get_coefs(Xs).squeeze()
        return coefs
        i_split = [self.pot_info.get_n_basis_2b()]  * 3 \
                + [self.pot_info.get_n_basis_eam()] * 3 \
                + [self.pot_info.get_n_basis_3b()]  * 5
        n_basis = sum(i_split)+i_split[-1]
        i_split = np.cumsum(i_split)
        i_swap = [0, 3, 1, 4, 2, 5] + list(range(6, 12))
        coefs = np.pad(coefs, (0, n_basis - coefs.shape[0]), 'constant')

        coefs = np.split(coefs, i_split)
        for i in list(range(6, 12)):
            coefs[i] *= 0.
        coefs = np.concatenate([coefs[i] for i in i_swap])
        return coefs
  
    def get_potential(self, obj):
        compound = self.get_compound(obj)
        try:
            pot = self.potentials[compound]
        except KeyError:
            pot = Potential(atoms_init=compound, **self.kwargs_pot)
            pot.init_regressor(coefs=self.get_coefs(compound))
            self.potentials.update({compound: pot}) 
            return pot
        
        try:
            pot.regressor.coefs * 1
        except:
            pot.init_regressor(coefs=self.get_coefs(compound))
            return pot

    def save_nn(self, path=None):
        self.nn.save_nn(path=path)
    
    def load_nn(self, path=None):
        self.nn.load_nn(path=path)
    
    def load_nn_weights(self, path=None, filename='weights'):
        self.nn.load_weights(path=path, filename=filename)
    
    def save_nn_weights(self, path=None, filename='weights'):
        self.nn.save_weights(path=path, filename=filename)
            
       
