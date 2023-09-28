'''
# -*- coding: utf-8 -*-
-------------------------------------------------------------------------------
PPGEET / UFF
Pedro A. Vieira
Data da criação:
Data da modificação:  
Versão:
-------------------------------------------------------------------------------

Descrição:
    Avaliação de pdf de melhor ajuste para amostras/dadso de entrada
    na saída fornece a pdf de melhor ajuste

'''
#%% Bibliotecas
# -------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm

#%% Função para ajuste de distribuições
# -------------------------------------
'''
Referência:
https://nedyoxall.github.io/fitting_all_of_scipys_distributions.html
'''

def fit_scipy_distributions(array,
                            bins,
                            plot_hist = True,
                            plot_best_fit = True,
                            plot_all_fits = False,
                            titulo = '',
                            eixoX = '',
                            eixoY = ''):
    """
    Fits a range of Scipy's distributions (see scipy.stats) against an array-like input.
    Returns the sum of squared error (SSE) between the fits and the actual distribution.
    Can also choose to plot the array's histogram along with the computed fits.
    N.B. Modify the "CHANGE IF REQUIRED" comments!
    
    Input: array - array-like input
           bins - number of bins wanted for the histogram
           plot_hist - boolean, whether you want to show the histogram
           plot_best_fit - boolean, whether you want to overlay the plot of the best fitting distribution
           plot_all_fits - boolean, whether you want to overlay ALL the fits (can be messy!)
    
    Returns: results - dataframe with SSE and distribution name, in ascending order (i.e. best fit first)
             best_name - string with the name of the best fitting distribution
             best_params - list with the parameters of the best fitting distribution.
    """
    
    if plot_best_fit or plot_all_fits:
        assert plot_hist, "plot_hist must be True if setting plot_best_fit or plot_all_fits to True"
    
    # Returns un-normalised (i.e. counts) histogram
    y, x = np.histogram(np.array(array), bins = bins) #, density = True)
    
    # Some details about the histogram
    bin_width = x[1]-x[0]
    N = len(array)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0 # go from bin edges to bin middles
    
    # selection of available distributions
    # CHANGE THIS IF REQUIRED
    DISTRIBUTIONS = [# st.alpha,
                     # st.cauchy,
                     # st.cosine,
                     # st.laplace,
                     # st.levy,
                     # st.levy_l,
                     st.norm,
                     # st.gamma,
                     # st.beta,
                     st.nakagami,
                     st.rayleigh,
                     st.rice,
                     st.lognorm,
                     # st.chi2
                     ]

    if plot_hist:
        fig, ax = plt.subplots()
        h = ax.hist(np.array(array), bins = bins, color = 'g') #, density = True)

    # loop through the distributions and store the sum of squared errors
    # so we know which one eventually will have the best fit
    sses = []
    for dist in tqdm(DISTRIBUTIONS):
        name = dist.__class__.__name__[:-4]

        params = dist.fit(np.array(array))
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
        pdf_scaled = pdf * bin_width * N # to go from pdf back to counts need to un-normalise the pdf
        '''
        ------------------------------------------
        Cálculo de erros podem ser colocados aqui:
        ------------------------------------------
        '''
        sse = np.sum((y - pdf_scaled)**2) # ===> Soma dos quadrado do erro - SSE
        sses.append([sse, name])
        '----------------------------------------'
        # Not strictly necessary to plot, but pretty patterns
        if plot_all_fits:
            ax.plot(x_mid, pdf_scaled, label = name)
    
    if plot_all_fits:
        plt.legend(loc=1)

    '''--------------------------
       # CHANGE THIS IF REQUIRED 
    -----------------------------'''
    ax.set_title(titulo)        # ('Curva de melhor ajuste')
    ax.set_xlabel(eixoX)        # ('x label')
    ax.set_ylabel(eixoY)        # ('y label')

    # Things to return - df of SSE and distribution name, the best distribution and its parameters
    results = pd.DataFrame(sses, columns = ['SSE','distribution']).sort_values(by='SSE') 
    best_name = results.iloc[0]['distribution']
    best_dist = getattr(st, best_name)
    best_params = best_dist.fit(np.array(array))
    
    if plot_best_fit:
        new_x = np.linspace(x_mid[0] - (bin_width * 2), x_mid[-1] + (bin_width * 2), 1000)
        best_pdf = best_dist.pdf(new_x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        best_pdf_scaled = best_pdf * bin_width * N
        ax.plot(new_x, best_pdf_scaled, label = best_name, c = 'r')
        plt.legend(loc=1)
    
    if plot_hist:
        plt.show()
    
    return results, best_name, best_params, \
            x_mid, pdf_scaled, new_x, best_pdf_scaled, best_pdf, x, y

# #%% Função para teste
# #--------------------
# def funcaoTesteAjuste():
#     '''
#     Teste da função:
#     ----------------
#     First we have to come up with a test array - I'll pick just a Gaussian with a few random parameters:
#     '''
#     test_array = st.norm.rvs(loc=7, scale=13, size=10000, random_state=0)
#     ''' 
#     Now we perform the fit with the function's standard settings. 
#     '''
#     sses, best_name, best_params = fit_scipy_distributions(test_array, bins = 100)
#     '''
#     (...) plotting all the distributions it is trying:
#     '''
#     sses, best_name, best_params = fit_scipy_distributions(test_array, 100, plot_best_fit=False, plot_all_fits=True)
#     #
#     return
#
