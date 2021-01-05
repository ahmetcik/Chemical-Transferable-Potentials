import numpy as np
from itertools import combinations_with_replacement
from scipy.special import sph_harm
from ortho_basis import OrthoBasis
import os
from mpl_toolkits.mplot3d import Axes3D

try:
    # matscipy's c implementation is 20 times faster than the one of ase
    from matscipy.neighbours import neighbour_list
    get_neighbour_list = neighbour_list
except:
    from ase.neighborlist import neighbor_list
    get_neighbour_list = neighbor_list

try:
    import julia
    jl = julia.Julia()
    path1 = os.path.join(os.path.dirname(__file__), 'fast_3b_envdesc.jl')
    path2 = os.path.join(os.path.dirname(__file__), 'fast_3b_strucdesc.jl')
    get_3b_from_2b_desc_julia = jl.include(path1)
    sum_environmental_to_structural_julia = jl.include(path2)
except:
    pass


class Descriptor(object):
    """Module for calculating environmental and structural 2b+3b descriptors.
    The environmental descriptors are pairwise distances r_ij and triplets
    of pairwise distances [r_ij, r_ik, r_jk]. 
    The structural descriptors are sums over basis functions wich take the environmental 
    descriptors as inputs. e.g. using the inverse polynomials x^-12 and x^-6 as basis 
    functions leads to a 2d structural descriptor [sum_ij r_ij^-12, sum_ij r_ij^-6]  
    and to a lennard-jones potential if that vector is mapped linearly onto the target energy. 
    Possible 2b basis functions are polynomials and gaussians. For the 3b part a product
    of polynomials (for the radial part) and spherical harmonics are used.

    The unit of the structural descriptor is per atom, this means also target energy must be
    per atom.

    In case pyjulia is installed, running with python-jl is one order of magnitude faster.

    Parameters
    ----------
    r_cut_2b, r_cut_3b: float
        Cutoff of the sphere around atom inside which the 2b and 3b 
        descriptors/environments are considered.

    cutoff_width_2b, cutoff_width_3b: float
        Width of cutoff function f_cut which pulls down basis function b smoothly, 
        e.g. in case of 2b: b(r_ij) -> b(r_ij) * f_cut(r_ij) 
                            for r_cut - cutoff_width < r_ij < r_cut.
    symmetric_3b: bool, default True
        Repeats each triplet with second and third element swapped. Needed for
        symmetric 3b potential, e.g. it is desirable for ineraction energy
        of atoms A, B and C that E(A, B, C) = E(A, C, B), with A being central atom.

    r_centers, sigmas: array
        Parameters that specify Gaussians/RBF basis set of 2b structural descriptor: 
        {exp(- 0.5 * (r - r_centers_i)^2 / sigmas_i**2)}.

    degrees: array
        Parameters that specify polynomial basis set of 2b structural descriptor: 
        {r^degrees_i}
   
    ortho: bool
        If true 2b basis functions are orthogonalized. That makes sense if a gradient
        based method is used to learn the linear coefficients because orthogonalziaton
        decorrelates the sums over basis vectors.
   
    ortho_min: float
        Specifies lower boundary of range over which 2b bases functions are orthogonalzied, 
        e.g scalar product <f, g> = \int_{ortho_min}^{r_cut} f(x) * g(x) dx.

    lmax, nmax: int
        Maximum l of spherical harmonics and maximum n (degree) of radial polynomials
        basis set of 3b structural descriptor.

    shift_3d: True, None or 3d array
        Shift that specifies center of 3b descriptor space, again for decorrelating 
        the sums over basis vectors:
        [(0 - shift[0],     r_cut_3b - shift[0]), 
         (0 - shift[0],     r_cut_3b - shift[0]),
         (0 - shift[0], 2 * r_cut_3b - shift[0])]
        If True, shift = [r_cut_3b / 2, r_cut_3b / 2, r_cut_3b] is used.

    cut_to_sym: bool
        If True, the third dimension of the 3b desriptor space is cut

    ortho_basis: object (optional)
        A prespecified OrthoBasis class passed for orthogonalizing 2b basis set.


 
    
    """
    def __init__(self, r_cut_2b=5., r_cut_3b=3., cutoff_width_2b=1., cutoff_width_3b=1., 
                 symmetric_3b=True,
                 r_centers=None, sigmas=None, degrees=None, ortho_2b=None, #2b struc descriptor
                 ortho_3b=None,
                 r_centers_3b=None, sigmas_3b=None,
                 degrees_eam=None, sigma_eam=3., ortho_eam=None,        #eam struc descriptor
                 lmax=1, nmax=3, shift_3d=True, cut_to_sym=False,     #3b struc descriptor
                 ortho_basis_2b=None, ortho_basis_eam=None, ortho_basis_3b=None,
                 fac=None):
        
        self.r_cut_2b = float(r_cut_2b)
        self.r_cut_3b = float(r_cut_3b)
        self.cutoff_width_2b = cutoff_width_2b
        self.cutoff_width_3b = cutoff_width_3b

        self.symmetric_3b = symmetric_3b

        self.r_centers_3b = r_centers_3b
        self.sigmas_3b = sigmas_3b

        self.r_centers = r_centers
        self.sigmas = sigmas
        self.degrees = degrees

        self.degrees_eam = degrees_eam
        self.sigma_eam = sigma_eam

        self.lmax = lmax
        self.nmax = nmax
        self.cut_to_sym = cut_to_sym
        self.shift_3d = self._get_shift_3d(shift_3d)
        
        # make list attributes numpy arrays
        self._make_array()
        
        self.fac = fac


        self._fit_ortho_basis(ortho_2b, ortho_basis_2b, 
                              ortho_eam, ortho_basis_eam, 
                              ortho_3b, ortho_basis_3b)
        
    def _make_array(self):
        for key in ['r_centers', 'sigmas', 'degrees', 'degrees_eam', 'shift_3d']:
            val = self.__dict__[key]
            if val is not None:
                self.__setattr__(key, np.array(val))

    def _get_shift_3d(self, shift_3d):
        """If shift_3d is not None or not an array 
           return (approx.) of 3d descriptor space."""
        if shift_3d is None or isinstance(shift_3d, (np.ndarray, list, tuple)):
            return shift_3d
        elif self.cut_to_sym:
            return 0.5 * self.r_cut_3b * np.ones(3)
        else:
            return 0.5 * self.r_cut_3b * np.array([1., 1., 2.])

    
    def _fit_ortho_basis(self, ortho_2b, ortho_basis_2b, ortho_eam, ortho_basis_eam, ortho_3b, ortho_basis_3b):
        if ortho_basis_2b is not None:
            self.ortho_basis_2b = ortho_basis_2b
        elif ortho_2b is not None:
            if ortho_2b[1] is None:
                ortho_2b[1] = self.r_cut_2b
            self.ortho_basis_2b = self._get_ortho_basis(interval=ortho_2b, 
                                                           degrees=self.degrees, 
                                                           r_centers=self.r_centers,
                                                           sigmas=self.sigmas)

        if ortho_basis_eam is not None:
            self.ortho_basis_eam = ortho_basis_eam
        elif ortho_eam is not None:
            self.ortho_basis_eam = self._get_ortho_basis(interval=ortho_eam, 
                                                            degrees=self.degrees_eam) 
        if ortho_basis_3b is not None:
            self.ortho_basis_3b = ortho_basis_3b
        elif ortho_3b is not None:
            if ortho_3b[1] is None:
                ortho_3b[1] = self.r_cut_3b
            self.ortho_basis_3b = self._get_ortho_basis(interval=ortho_3b, 
                                                            degrees=np.arange(0, self.nmax+1)) 



    def _get_ortho_basis(self, interval=None, degrees=None, r_centers=None, sigmas=None):
        """Set transformation for orthogonalizing 2b basis set."""
        v_list = []

        if degrees is not None:
            for d in degrees:
                def f(x, d=d):
                    return x**d
                v_list.append(f)

        if r_centers is not None:
            for center, sigma in zip(r_centers, sigmas):
                def f(x, c=center, s=sigma):
                    return np.exp(-0.5 * (x - c)**2 / s**2)
                v_list.append(f)
           
        ortho_basis = OrthoBasis(interval=interval)
        ortho_basis.fit(v_list)
        return ortho_basis




    def init_atomic_numbers(self, atoms):
        """Initialize atomic number combinations. The 2b and 3b tuples are sorted by well defined rule, e.g.
           [(1, 1), (1, 2), (2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 2)].
        """
        try:
            atomic_numbers_unique = sorted(set(atoms.numbers))
        except:
            try:
                atomic_numbers_unique = sorted(set(np.concatenate([at.numbers for at in atoms])))
            except:
                atomic_numbers_unique = sorted(set(atoms))
        self.atomic_numbers  = [comb for comb in combinations_with_replacement(atomic_numbers_unique, 2)]
        self.i_split = len(self.atomic_numbers)
        self.atomic_numbers += [(z,) + tup for z in atomic_numbers_unique for tup in self.atomic_numbers]

    def get_environmental_descriptor(self, atoms, only_2b=False):
        """Desriptor of local environment.
        For 2b: r_ij, pairwise distance
        For 3b: (r_ij, r_ik, r_jk)
        """
        r_cuts = [self.r_cut_2b, self.r_cut_3b]
        i_max = np.argmax(r_cuts)
        r_max = r_cuts[i_max]
        
        # get pairwise distances d, corresponding atom indices i and j 
        # and positons vector diffs D for both 2b and 3b
        (i2, j2, d2), (i3, j3, d3, D3) = self._get_neighbours(atoms)
        

        
        ##### 2b
        ij = np.sort(atoms.numbers[np.transpose([i2, j2])], axis=1)
        desc_2b = self._split_according_to_atomic_numbers(d2[:, np.newaxis], ij, 
                                                          self.atomic_numbers[:self.i_split],
                                                          i=i2)
        ##### 3b
        if i3.size == 0 or only_2b:
            desc_3b = {atnu: {'desc': np.empty((0, 1, 3))}
                       for atnu in self.atomic_numbers[self.i_split:]}
        else:
            try:
                ds = get_3b_from_2b_desc_julia(i3, j3, d3, D3, atoms.numbers)
                atomic_numbers = self.atomic_numbers[self.i_split:]
                desc_3b = {atnu: {'desc': ds[i].T[:, np.newaxis, :]} 
                           for i, atnu in enumerate(atomic_numbers)}
            except:
                i3, j3, k3, d3 = self._get_3b_from_2b_desc(i3, j3, d3, D3, atoms.numbers)
                
                # sort only 2nd and 3rd column as descriptor symmetric in 2nd and 3rd entry
                ijk = np.column_stack((atoms.numbers[i3], np.sort(np.transpose([atoms.numbers[j3], atoms.numbers[k3]]))))
                desc_3b = self._split_according_to_atomic_numbers(d3[:, np.newaxis, :], ijk, 
                                                                  self.atomic_numbers[self.i_split:])

            
            if self.symmetric_3b:
                # in order to make 3b symmetric in column 0 and 1 add itself swapped in 0 and 1
                desc_3b  = {comb: {'desc': np.vstack([d['desc'], d['desc'][:, :, [1, 0, 2]]])}
                                   for comb, d in desc_3b.items()}
        return {**desc_2b, **desc_3b}

    def _get_neighbours(self, atoms):
        if self.fac is None:
            r_cuts = [self.r_cut_2b, self.r_cut_3b]
        else:
            r_cuts = [self.r_cut_2b*self.fac, self.r_cut_3b*self.fac]
        i_min, i_max = np.argsort(r_cuts)
        
        i, j, d, D = get_neighbour_list('ijdD', atoms, r_cuts[i_max])
        if self.fac is None:
            out = [(i, j, d), (i, j, d, D)]
        else:
            out = [(i, j, d/self.fac), (i, j, d/self.fac, D/self.fac)]
        if self.r_cut_2b != self.r_cut_3b:
            mask = d < r_cuts[i_min]
            out[i_min] = [x[mask] for x in out[i_min]]
        return out

    def _split_according_to_atomic_numbers(self, d, ijk, atomic_numbers, i=None):
        ijk_mask_dict = {comb: (ijk == comb).all(axis=1) for comb in atomic_numbers}
        desc = {comb: {'desc': d[mask]} for comb, mask in ijk_mask_dict.items()}
        if i is not None:
           desc = self._add_split_i_atom(desc, ijk_mask_dict, i)
        return desc
    
    def _add_split_i_atom(self, desc, ijk_mask_dict, i):
        for comb, mask in ijk_mask_dict.items():
            i_mask = i[mask]
            desc[comb]['i_atom'] = np.array([np.where(i_mask == i_atom)[0] for i_atom in np.unique(i_mask)])
        return desc

    def _get_3b_from_2b_desc(self, i, j, d, D, atoms_numbers):
        i, i_j, i_k = self._get_triplet_indices(i, len(atoms_numbers))    
        
        d_jk = np.linalg.norm(D[i_j] - D[i_k], axis=1)
        d = np.transpose([d[i_j], d[i_k], d_jk])
        
        j, k = j[i_j], j[i_k]
        return i, j, k, d

    def _get_triplet_indices(self, i, len_atoms):
        
        bincount = np.bincount(i)
        n_rep = bincount * (bincount-1) // 2
        i_new = np.repeat(np.arange(len_atoms), n_rep)
        
        i_j, i_k = [], []
        for i_atom in range(len_atoms):
            indices_neighbors = np.where(i== i_atom)[0]
            ji, ki = np.triu_indices(indices_neighbors.size, k=1)
            i_j.append(indices_neighbors[ji])
            i_k.append(indices_neighbors[ki])
        i_j = np.concatenate(i_j)
        i_k = np.concatenate(i_k)
        return i_new, i_j, i_k


    def _get_Yml_complex(self, l, m, theta, phi):
        """Wikipedia convention
        phi: [0, 2*pi]
        theta: [0, pi]"""
        return sph_harm(m, l, phi, theta)

    def _get_angular_basis(self, theta, phi, lmax=3):
        L, M = np.array([(l, m) for l in range(lmax+1) for m in range(-l, l+1)]).T
        Ylm = np.zeros((len(theta), len(L)))

        indices_0   = np.where(M == 0)[0]
        indices_pos = np.where(M >  0)[0]
        indices_neg = np.where(M <  0)[0]

        Ylm[:, indices_0  ] = np.real(self._get_Yml_complex(L[indices_0],    M[indices_0  ], theta[..., np.newaxis], phi[..., np.newaxis]))
        Ylm[:, indices_pos] = np.real(self._get_Yml_complex(L[indices_pos],  M[indices_pos], theta[..., np.newaxis], phi[..., np.newaxis])) * np.sqrt(2.) * (-1.)**M[indices_pos]
        Ylm[:, indices_neg] = np.imag(self._get_Yml_complex(L[indices_neg], -M[indices_neg], theta[..., np.newaxis], phi[..., np.newaxis])) * np.sqrt(2.) * (-1.)**M[indices_neg]
        return Ylm

    def _get_spherical_coordinates(self, x, y, z): 
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        phi = np.arctan2(y, x)
        return theta, phi 

    def sum_environmental_to_structural(self, desc, n_body=2, n_atoms=1., return_nosum=False, is_eam=True, not_julia=False):
        """Sum environmental descriptors up to structural descriptors using
        defined basis sets, e.g. Gaussians.
        For example, for 2b: s_k = sum_ij exp(-0.5 * (r_ij - r_k)**2 / sigma_k**2)
        for different centers and sigmas.

        Returns array of sums where array is n_basis dimensional.
        """
        r = desc['desc'].copy()
        if n_body == 2:
            r_cut = self.r_cut_2b
            cutoff_width = self.cutoff_width_2b
        else:
            r_cut = self.r_cut_3b
            cutoff_width = self.cutoff_width_3b
            if not not_julia:
                try:
                    g = sum_environmental_to_structural_julia(r, r_cut, cutoff_width, 
                                                              self.r_centers_3b, self.sigmas_3b)
                    return g / n_atoms
                except:
                    pass


        indices_for_smooth_cutoff, f_cut = self._get_f_cut(r, r_cut=r_cut, cutoff_width=cutoff_width, n_body=n_body)
        
        if n_body == 2:
            basis_output = self._get_basis_output_2b(r)
        else:
            basis_output = self._get_gaus3b(r)
            #basis_output = self._get_radial_3b(r)
        
        # multuply basis function outputs with cutoff function
        basis_output[indices_for_smooth_cutoff] *= f_cut

#        if n_body == 3:
#            basis_output = self._add_spherical_3b(r, basis_output)

        if return_nosum:
            return basis_output / n_atoms
        else:
            basis_sum = basis_output.sum(0) 
            
            if n_body == 2 and is_eam:
                eam_sum = self._get_eam_sum(r, indices_for_smooth_cutoff, f_cut, desc['i_atom'])
                basis_sum = np.append(basis_sum, eam_sum)
            return basis_sum / n_atoms
    
    def _get_eam_sum(self, r, indices_for_smooth_cutoff, f_cut, i_atom):
        basis_output = np.exp( - 2. * r / self.sigma_eam + 2.)
        basis_output[indices_for_smooth_cutoff] *= f_cut

        eam = np.array([np.sqrt(basis_output[indices].sum())**self.degrees_eam 
                        for indices in i_atom])

        try:
            eam = self.ortho_basis_eam.transform(eam)
        except:
            pass
        
        if eam.size == 0:
            return np.zeros(self.degrees_eam.size)
        else:
            return eam.sum(0)

    def _get_basis_output_2b(self, r):
        basis_output = np.empty((r.shape[0], 0))
        if self.degrees is not None:
            polynomials = r**self.degrees
            basis_output = np.hstack((basis_output, polynomials))

        if self.r_centers is not None:
            gaussians = np.exp( - 0.5 * (r - self.r_centers)**2 / self.sigmas**2)
            basis_output = np.hstack((basis_output, gaussians))
        try:
            basis_output = self.ortho_basis_2b.transform(basis_output)
        except:
            pass
        return basis_output

    def _get_gaus3b(self, r):
        diff = np.linalg.norm(r - self.r_centers_3b, axis=2)/self.sigmas_3b
        gaussians = np.exp( - 0.5 * diff**2 )
        return gaussians


    def _get_radial_3b(self, r):
        if self.centers_3b is not None:
            return self._get_gaus3b(r)
        if self.shift_3d is not None:
             r -= self.shift_3d
        r_radial = np.linalg.norm(r, axis=2)
        degrees = np.arange(0, self.nmax+1)
        basis_output = r_radial**degrees
        try:
            basis_output = self.ortho_basis_3b.transform(basis_output)
        except:
            pass
        return basis_output
            
    def _add_spherical_3b(self, r, basis_output):
        if self.centers_3b is not None:
            return basis_output
        theta, phi = self._get_spherical_coordinates(*r.squeeze().T)
        Ylm = self._get_angular_basis(theta, phi, lmax=self.lmax)
        return np.hstack([basis_output * Ylm[:, [i]] for i in range(Ylm.shape[1])])

    def _cos(self, r, r_cut=5., cutoff_width=.5):
        return 0.5 * (np.cos(np.pi / cutoff_width * (r - r_cut + cutoff_width)) +1.)

    def _get_f_cut(self, r, r_cut=5., cutoff_width=1., n_body=2):
        """Returns cutoff value and indices on which cutoff needs to be applied.
        If self.cut_to_sym, r_jk of 3b descriptor [r_ij, r_ik, r_jk] is cut
        at r_cut sucht that descriptor space is symmetric. Otherwise max value
        for r_jk is around 2*r_cut.
        """
        if cutoff_width > 0.: 
            if n_body == 2:
                mask_for_smooth_cutoff = (r > r_cut - cutoff_width).squeeze()
                f_cut = self._cos(r[mask_for_smooth_cutoff], r_cut=r_cut, cutoff_width=cutoff_width)
            else:
                r = r.squeeze()
                if self.cut_to_sym:
                    mask_zero = r[:, [-1]] >= r_cut
                    mask_for_smooth_cutoff_3d = (r > r_cut - cutoff_width) & (r < r_cut) & ~mask_zero
                    mask_for_smooth_cutoff = np.any(mask_for_smooth_cutoff_3d, axis=1) | mask_zero.squeeze()
                else:
                    r = r[:, :2]
                    mask_for_smooth_cutoff_3d = r > r_cut - cutoff_width
                    mask_for_smooth_cutoff = np.any(mask_for_smooth_cutoff_3d, axis=1)

                f_cut = np.ones_like(r)
                f_cut[mask_for_smooth_cutoff_3d] = self._cos(r[mask_for_smooth_cutoff_3d], r_cut=r_cut, cutoff_width=cutoff_width)
                
                if self.cut_to_sym:
                    f_cut[mask_zero.squeeze()] = 0
                
                f_cut = f_cut[mask_for_smooth_cutoff].prod(1)[:, np.newaxis]
        else:
            indices_for_smooth_cutoff = []
            f_cut = 1.
        return mask_for_smooth_cutoff, f_cut
   
    def get_structural_descriptor(self, atoms, n_atoms=None):
        """Calc sums of descriptors for each atomic number combination, e.g. 2b and 3b.
        For example, for 2b: s_k = sum_ij exp(-0.5 * (r_ij - r_k)**2 / sigma_k**2)
        for different centers and sigmas. Concatenating the arrays of all atomic number 
        combinations results in a structural descriptor array. In this framework
        the structural descriptor is mapped linearly onto the target energy

        Either atoms object or precalculated descriptor dict can be passed, 
        the latter only in combination with kwarg n_atoms

        Returns a dict with atomic number pairs and triplets as keys and
        structural descriptors as values.
        """
        if isinstance(atoms, dict):
            desc = atoms
            try: 
                int(n_atoms)
            except:
                raise ValueError("If descriptor dict is passed to get_structural_descriptor \
                                  also kwarg n_atoms needs to be specified, \
                                  e.g. get_structural_descriptor(desc, n_atoms=2).")
        else:
            desc = self.get_environmental_descriptor(atoms)
            n_atoms = len(atoms)
        structural_descripor_dict = {comb: self.sum_environmental_to_structural(desc[comb], n_body=len(comb),
                                     n_atoms=n_atoms) for comb in self.atomic_numbers}
        return structural_descripor_dict


    def get_nearest_neighbour_distance(self, atoms):
        """Returns nearest neighbour distance in crystal."""
        diff = atoms.positions[:, np.newaxis, :] - atoms.positions
        diff = np.linalg.norm(diff, axis=2)
        d_closest = min(diff[np.triu_indices(diff.shape[0], k=1)])

        # in some cases closest is in image, therefore:
        d_closest = get_neighbour_list('d', atoms, d_closest+0.01).min()
        return d_closest
    
    def get_n_basis_2b(self):
        n_basis = 0 
        if self.r_centers is not None:
            n_basis += len(self.r_centers)
        if self.degrees is not None:
            n_basis += len(self.degrees)
        return n_basis

    def get_n_basis_eam(self):
        return len(self.degrees_eam)
    
    def get_n_basis_3b(self):
        return self.sigmas_3b.size




