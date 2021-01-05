from ctp import CTP
from util.load_data import *
from util.plot import *
from util.utilities import *

import json
import pandas as pd

from ase.spacegroup import crystal, get_spacegroup
from ase.visualize import view
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS

C = get_chemical_informaton("data/atomic_features.csv")
df = get_data("data/structures.xyz")
Y = df['energy'].values

ctp_parameters = json.load(open('data/ctp_parameters.json'))

r_centers, sigmas = get_3b_centers(2., 6., steps=4, sigma=1.5)
ctp_parameters['r_centers_3b'] = r_centers
ctp_parameters['sigmas_3b'] = sigmas

ctp_parameters['atomic_info'] = C

Rat = pd.read_csv('data/Rat.csv', index_col='AB')

if True:
    df['AB'] = df['A'] + df['B']
    ABs = df['AB'].drop_duplicates().values
    X = []
    for i,ab in enumerate(ABs[:]):
        fac = Rat.loc[ab, 'rat_pred']
        print('#########', i, ab, fac)
        ctp = CTP(fac=fac, **ctp_parameters)
        x = ctp.get_X(df[df['AB'] == ab]['atoms'].values)
        X.append(x)
    X =np.vstack(X)
    #np.savetxt('tmp/X.dat', X)
else:
    X = np.loadtxt('tmp/X.dat', X)

leave_out_ABs = list(np.loadtxt('data/leave_out.dat', dtype=str))
leave_out_ABs, ABS_vali = leave_out_ABs[:1], leave_out_ABs[1:]
X_train, Y_train, X_test, Y_test, X_vali, Y_vali = split_data(df, X, leave_out_ABs, ABS_vali)


nn_train_parameters = {'optimizer_type': 'adam',
                       'learning_rate': 0.001,
                       'n_epochs': 10,
                       'stopping_threshold': 0.0,
                       'batch_size': 32,
                       'lambda_reg': 0.,
                       'keep_2b': 1.,
                       'keep_3b': 1.,
                       'epoch_save_weights': None,
                       'decay': None,
                       'epoch_start_3b': 0,
                      }

ctp = CTP(**ctp_parameters)
ctp.set_nn_train_para(**nn_train_parameters)
ctp.fit(X_train, Y_train, vali=[X_vali, Y_vali])

ctp.load_nn_weights(filename='weights_best')
ctp.save_nn()
 
Y_test_pred  = ctp.predict(X_test)
print(abs(Y_test_pred, Y_test).mean())





