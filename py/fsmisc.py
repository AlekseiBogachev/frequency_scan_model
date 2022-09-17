import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fsmodels import SklSingleExpFrequencyScan


def plot_spectr(exps_params_, xlim=[1/2500, 1], ylim=None):
    fig, ax = plt.subplots(1,1)

    exps_params = np.c_[np.power(10.0, exps_params_[:, 0]), 
                        exps_params_[:, 1]]
    
    for TC, AMP in exps_params:
        ax.semilogx([TC, TC], [0, AMP], '-b')
        
    ax.set_title('Спектр')
    ax.set_xlabel('Постоянная времени, с')
    ax.set_ylabel('Амплитуда')
    ax.grid()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    return fig, ax


def _get_y(X, model_class, **model_params):
    
    if 'n_exps' in model_params.keys():
        model = model_class(filling_pulse = model_params['filling_pulse'],
                            n_exps = model_params['n_exps'])
    else:
        model = model_class(filling_pulse = model_params['filling_pulse'])

    model.exps_params_ = model_params['exps_params_']
    
    if hasattr(model_class, 'p_coef_'):
        model.p_coef_ = model_params['p_coef_']
    
    return model.predict(X)


def _get_params_fom_iter(iter, fit_results_):
    
    selected_row = fit_results_.iloc[iter, :]

    n_exps = selected_row['n_exps']

    col_pairs = [[f'time_constant_pow_{i}', f'amplitude_{i}'] 
                 for i in range(n_exps)]
                 
    exps_params_ = np.stack([selected_row[pair].to_numpy() for pair in col_pairs])

    model_params = dict(filling_pulse = selected_row['filling_pulse'],
                        exps_params_ = exps_params_)
    
    if 'p_coef' in fit_results_:
        model_params['p_coef_'] = selected_row['p_coef']
    else:
        model_params['n_exps'] = n_exps
    
    return model_params


def plot_perfect_scan(X, exps_params_, f_pulse, label, marker='-.'):
    fs = SklSingleExpFrequencyScan(filling_pulse = f_pulse)
    fs.exps_params_ = exps_params_
    fs.p_coef_ = 1.0
    y = fs.predict(X)
    plt.plot(X, y, marker, label=label)


def plot_model(X, y, model_class, fit_results_, plot_exps=True):

    initial_params = _get_params_fom_iter(0, fit_results_)
    initial_dlts = _get_y(X, model_class, **initial_params)

    final_params = _get_params_fom_iter(-1, fit_results_)
    final_dlts = _get_y(X, model_class, **final_params)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.plot(X, y, 'og', alpha=0.3, label='Экспериментальные данные')
    ax.plot(X, initial_dlts, '-b', label='Начальная модель')
    ax.plot(X, final_dlts, '-r', label='Модель после идентификации')

    if plot_exps:
        for i, params in enumerate(final_params['exps_params_']):
            plot_perfect_scan(X, 
                              [params], 
                              final_params['filling_pulse'], 
                              label=f'exp{i}'
                             )

    ax.legend()
    ax.grid()
    ax.set_xlim([0, 3.5])
    ylim = ax.get_ylim()
    ylim = [np.floor(ylim[0]), np.ceil(ylim[1])]
    ax.set_ylim(ylim)
    
    ax.set_title('Результаты идентификации')
    ax.set_xlabel(r'$\log_{10}(F_0), \log_{10}(Гц)$')
    ax.set_ylabel('Сигнал DLTS, условные единицы')
    
    return fig, ax