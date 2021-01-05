import pandas as pd
import numpy as np
import ase.io
from ase.data import chemical_symbols, atomic_numbers
from pymatgen.core.periodic_table import Element

def get_chemical_symbols_binary(atoms):
    cs = [chemical_symbols[n] for n in sorted(set(atoms.numbers))]
    if len(cs) == 1:
        cs *= 2
    return cs

def get_data(path):
    atoms_list = ase.io.read(path, ':')
    data = [(*get_chemical_symbols_binary(at), at.info['structure_type'],
              at.info['energy'], at.get_volume()/len(at), at)
              for at in atoms_list]
    df = pd.DataFrame(data, columns=['A', 'B', 'structure_type', 'energy', 'volume', 'atoms'])
    return df.sort_values(['A', 'B', 'structure_type', 'volume']).reset_index(drop=True)


def get_dimer_distance(at):
    return abs(at.positions[0,0] - at.positions[1,0])

def get_dimers_data(path, max_energy=None, r_cut=None):
    atoms_list = ase.io.read(path, ':')
    data = [(*get_chemical_symbols_binary(at), at.info['energy'], get_dimer_distance(at), at,)
              for at in atoms_list]
    df = pd.DataFrame(data, columns=['A', 'B', 'energy', 'distance', 'atoms'])

    if max_energy is not None:
        df = df[df['energy'] < max_energy]

    if r_cut is not None:
        df = df[df['distance'] < r_cut]
        df_last = df.sort_values('distance').groupby(['A', 'B'], as_index=False).last()
        df = df.merge(df_last[['A', 'B','energy']], on=['A', 'B'], suffixes=('', '_last'))
        df['energy'] -= df['energy_last']
        df = df.drop('energy_last', axis=1)
    return df.reset_index(drop=True)

def get_chemical_informaton(path, features=['Z', 'Group', 'Row', 'IP', 'EA', 'r_s', 'r_p']):
    df = pd.read_csv(path)

    C = df[features].values
    
    # first column needs to be atomic number (and may also be a float)
    C = np.column_stack((df['Z'].values, C))
    return C

def split_data(df, X, leave_out_ABs, ABS_vali, remove_ABs=[]):

    Y = df['energy'].values
    
    ABs  = (df['A'] + df['B']).values
    ABSs = (df['A'] + df['B'] + df['structure_type']).values
    #ABs = get(df[['A', 'B']].values)
    #ABSs = [ABs[i]+S for i, S in enumerate(df['structure_type'].values)]

    indices_test   = [i for i, AB  in enumerate(ABs)  if AB  in leave_out_ABs]
    indices_remove = [i for i, AB  in enumerate(ABSs)  if AB  in remove_ABs]
    indices_vali   = [i for i, ABS in enumerate(ABSs) if ABS in ABS_vali]
    indices_train  = sorted(set(range(len(ABs))) - set(indices_test) -  set(indices_remove) - set(indices_vali))

    X_train = X[indices_train]
    Y_train = Y[indices_train]
    X_test  = X[indices_test]
    Y_test  = Y[indices_test]
    X_vali  = X[indices_vali]
    Y_vali  = Y[indices_vali]
    return X_train, Y_train, X_test, Y_test, X_vali, Y_vali

def get(AB):
    gA = [Element(A).group if Element(A).group != 14 else 1 for A in AB[:, 0]]
    gB = [Element(B).group if Element(B).group != 14 else 0 for B in AB[:, 1]]
    gAB = np.transpose([gA, gB])
    I = np.argsort(gAB, axis=1)
    AB_sum = [AB[i][I[i][0]]+AB[i][I[i][1]] for i in range(len(gA))]
    return AB_sum 
    

