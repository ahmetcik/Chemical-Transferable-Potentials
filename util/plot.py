import numpy as np
import matplotlib.pyplot as plt
from processing import get_rmse
from ase.data import chemical_symbols
from itertools import product

def plot_scatter(x, y, save_path=None, markersize=7, markeredgewidth=0.5, markeredgecolor='k', xlabel='Reference', ylabel='Prediction'):
    
    rmse = get_rmse(x, y)
    mae = abs(x - y).mean()
    mini = min([min(x), min(y)])
    maxi = max([max(x), max(y)])
    
    plt.plot([mini, maxi], [mini, maxi], 'k-')
    plt.plot(x, y, 'o', markersize=markersize, markeredgecolor=markeredgecolor, markeredgewidth=markeredgewidth)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('RMSE: %.3f   MAE: %.3f' % (rmse, mae))

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.cla()
    plt.close()

def plot_pes(y_ref, y_pred, x=None, save_path=None):
    if x is None:
        x = np.arange(y_ref.size)

    rmse = get_rmse(y_ref, y_pred)
    mae = abs(y_ref - y_pred).mean()

    plt.plot(x, y_ref,  's-', label='Ref')
    plt.plot(x, y_pred, 's-', label='Pred')
    plt.title('RMSE: %.3f   MAE: %.3f' % (rmse, mae))
    plt.legend()
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.cla()
    plt.close()


def plot_train_test(train_ref, train_pred, test_ref, test_pred, save_path=None, title=None):
    
    rmse_train = get_rmse(train_ref, train_pred)
    rmse_test  = get_rmse(test_ref,  test_pred)
    mae_train  = abs(train_ref - train_pred).mean()
    mae_test   = abs(test_ref  -  test_pred).mean()


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,10))
    
    ax1.plot(train_ref)
    ax1.plot(train_pred)
    ax1.set_title('RMSE: %.3f   MAE: %.3f' % (rmse_train, mae_train))
    
    ax2.plot(test_ref, 's-', label='Ref')
    ax2.plot(test_pred, 's-', label='Pred')
    ax2.set_title('RMSE: %.3f   MAE: %.3f' % (rmse_test, mae_test))
    
    if title is not None:
        f.suptitle(title, fontsize=14)

    plt.legend()
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.cla()
    plt.close()

def plot_all_pes(df, save_path='Separate_pes', x=None):
    df['AB'] = df['A'] + df['B']
    AB_unique = sorted(set(df['AB'].values))

    if x is None:
        x_AB = x

    for i, AB in enumerate(AB_unique):
        df_AB = df[df['AB'] == AB]
        y_ref  = df_AB['energy'].values
        y_pred = df_AB['energy_pred'].values
        path = "%s/%s_%s.png" % (save_path, i, AB)

        if x is not None:
            x_AB = df_AB[x].values
        plot_pes(y_ref, y_pred, save_path=path, x=x_AB)



def plot_coefs(ctp, compounds, features=None, n_points=100, save_path=None):
    Xs = ctp.get_chemical_input(np.unique(compounds, axis=0))
    X = Xs[1]
    
    X_mean = np.repeat(X.mean(0)[np.newaxis], n_points, axis=0)
    X_path = np.linspace(X.min(0), X.max(0), n_points)
    
    n_coef = ctp.pot_info.get_n_basis_2b()
    n_feat = X.shape[1]
    
    if features is None:
        features = ['feat_%s' %i  for i in range(n_feat)]

    fig, axes = plt.subplots(n_coef, n_feat, figsize=(30,30))
    
    for idx, (i, j) in enumerate(product(range(n_feat), range(n_coef))):
        if i == 0:
            axes[j][i].set_ylabel('coef_%s' % j)
        
        if j == 0:
            X_plot = X_mean.copy()
            X_plot[:, i] = X_path[:, i]
            coefs_matrix = ctp.nn.get_coefs_pair(X_plot)
            axes[j][i].set_title(features[i])
        
        axes[j][i].plot(X_plot[:, i], coefs_matrix[:, j])
               
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.cla()
    plt.close()


