'''
# -*- coding: utf-8 -*-
----------------------------------------------------------------------------
PPGEET / UFF
Disciplina:
Prof.:
Aluno: Pedro A. Vieira
Data da criação:
Data da modificação:  
Versão:
Descrisão:
'''
#
#%% Bibliotecas
'------------------------------------------------------------------------------'
#
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import os
import time

from scipy import signal

from distfit import distfit

# Entrelaçar os dados
from toolz import interleave

from sklearn.utils import resample
from sklearn.metrics import r2_score, \
                            mean_absolute_error, \
                            mean_squared_error, \
                            mean_squared_log_error, \
                            explained_variance_score
#
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split, \
                                    GridSearchCV, \
                                    RandomizedSearchCV, \
                                    cross_val_score
#
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor # ANN
import joblib
#
import pickle
#
# from ann_visualizer.visualize import ann_viz;
#
import seaborn as sns

#
#%% Definição de funções
'------------------------------------------------------------------------------'

#
#%% Leitura do arquivo Excel para treinamento do modelo
'------------------------------------------------------------------------------'
#
# Formação de dicionário e dataframe com dados da medição
# https://stackoverflow.com/questions/66947749/create-pandas-dataframe-with-different-sized-columns
medicoes = {'rota': [1, 2],
            'altura_antena': [ 'H1', 'H2'],
            'frequencia': ['F1', 'F2', 'F3', 'F4', 'F5']}

medicoes_df = pd.DataFrame.from_dict(medicoes, orient = 'index').T

medicoes_rota_01 = ['R1H1F1', 'R1H1F2', 'R1H1F3', 'R1H1F4', 'R1H1F5',
                    'R1H2F1', 'R1H2F2', 'R1H2F3', 'R1H2F4', 'R1H2F5']
#
medicoes_rota_02 = ['R2H1F1', 'R2H1F2', 'R2H1F3', 'R2H1F4', 'R2H1F5',
                    'R2H2F1', 'R2H2F2', 'R2H2F3', 'R2H2F4', 'R2H2F5']
#
caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/"
#
# Nome do arquivo a ser processado
'''----------------------------------------'''
# Exemplo: R1H1F1_df_proc.xlsx
ext_arq = ".xlsx" # extensao do arquivo 

comp_nome_1 = '_df_proc_1'
comp_nome_2 = '_df_proc_2'
comp_nome_3 = '_df_proc_3'

# comp_nome_1 = '_df_proc'
# comp_nome_2 = '_saida'
# comp_nome_3 = '_df_proc_saida'
# comp_nome_4 = '_df_proc_3'
#
''' Planilhas Rota 01
--------------------- '''
for conta_med in medicoes_rota_01:
    #
    conta_med_proc = conta_med + comp_nome_3 # R1H1F1_df_proc_3
    # conta_med_proc_1_2 = conta_med + comp_nome_4
    #
    print('-->> Lendo ' + conta_med_proc + ext_arq)
    locals()[conta_med] = pd.ExcelFile(caminho + \
                                              conta_med_proc + \
                                              ext_arq).parse(sheet_name = conta_med)
#
''' Planilhas Rota 02
--------------------- '''
for conta_med in medicoes_rota_02:
    #
    conta_med_proc = conta_med + comp_nome_3 # R1H1F1_df_proc
    # conta_med_proc_1_2 = conta_med + comp_nome_4
    #
    print('-->> Lendo ' + conta_med_proc + ext_arq)
    locals()[conta_med] = pd.ExcelFile(caminho + \
                                              conta_med_proc + \
                                              ext_arq).parse(sheet_name = conta_med)
#
#%% Obtem lista com o cabeçalho das colunas
'------------------------------------------------------------------------------'
#
cab_colunas = locals()[medicoes_rota_01[0]].columns.tolist()
#
medicoes_rota = medicoes_rota_01 + medicoes_rota_02
#
cab_colunas_rota_01 = locals()[medicoes_rota_01[0]].columns.tolist()
cab_colunas_rota_02 = locals()[medicoes_rota_02[0]].columns.tolist()
#
#%% Escolha da rota de processamento
'------------------------------------------------------------------------------'
#
controla_escolha = 0
while controla_escolha==0:
    #
    nome_rota_proc = input("\n\nEntre com a rota para processamento: [1] / 2: \n")
    if len(nome_rota_proc)==0 or nome_rota_proc == '1':
        nome_rota_proc = '1'
        medicoes_rota = medicoes_rota_01
        # cab_colunas = cab_colunas_rota_01
        controla_escolha = 1
    elif nome_rota_proc == '2':
        medicoes_rota = medicoes_rota_02
        # cab_colunas = cab_colunas_rota_02
        controla_escolha = 1
    else:
        nome_rota_proc = ''
        controla_escolha = 0
#
#%% Leitura arquivos dados do sistema
'------------------------------------------------------------------------------'
#
caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/Medicoes/"
nome_arq_dados_sistema = 'DadosDoSistema_v1.xlsx'
nome_planilha = 'Planilha1'

dados_sistema = \
    pd.ExcelFile(caminho + nome_arq_dados_sistema).parse(sheet_name = nome_planilha)
#
#%% Dados do sistema para variáveis
'------------------------------------------------------------------------------'
#
cab_dados_sist = dados_sistema.columns.tolist()
#
valor_pos_cab = 0
for conta_cab in cab_dados_sist:
    #
    # locals()[conta_cab] = dados_sistema[conta_cab].values
    if valor_pos_cab == 0:
        valor_pos_cab = valor_pos_cab + 1
        pass
    else:
        locals()[conta_cab] = float(dados_sistema.loc[0,conta_cab])
        # print(conta_cab)
#
#%%
''' ---------------------------------------------------- '''
ENTER = input('\nPressione [ENTER] para continuar...\n')
''' ---------------------------------------------------- '''
#
# %% pdf - Teste de verificação de pdf de melhor ajuste
'------------------------------------------------------------------------------'
#
# Importação da função
'---------------------'
import melhorAjuste_pdf_v0 as fit_pdf
#
# Cabeçalho
# ---------
escolha_col = ''
conta_pos_col = 0
for element in cab_colunas_rota_01:
    print(str(conta_pos_col) + ' - ' + element)
    conta_pos_col +=1
#
# Esolha dos dados
# ----------------------------------------
escolha = ''
while not escolha:
    #
    # Escolha das variáveis de entrada
    # --------------------------------
    var_entrada=[]
    while True:
        lista = input('\n Entre número da posição da variável de entrada: ');
        if len(lista) == 0:
            break
        var_entrada.append(cab_colunas_rota_01[int(lista)])
    #
    print('\n---------------------------------------------------------------------')
    print('Variáveis de entrada: ')
    print(var_entrada)
    print('---------------------------------------------------------------------')
    
    #
    sinal_avaliado =  locals()[conta_med][var_entrada].values
    
    sinal_avaliado = sinal_avaliado.reshape(-1,1)
    sinal_avaliado = sinal_avaliado[~np.isnan(sinal_avaliado).any(axis=1)] # Eliminação de linha com nan
    
    plt.plot(sinal_avaliado)
    plt.grid()
    plt.show()

    escolha = input('\n nova escolha [s]/n ?')
    if len(escolha) == 0 or escolha == 's':
        escolha = ''
    else:
        escolha = 'n'
# %% Lopp para avaliar as rotas
'------------------------------'
#
'--------------------------------'
teste_medicao = medicoes_rota
# Comentar a linha abaixo se não for utilizar
teste_medicao = ['R1H2F3'] # Se especificado executa somente o indicado
'--------------------------------'
#
controle_tabela = 0
for conta_med in teste_medicao:

    sinal_avaliado =  locals()[conta_med][var_entrada].values
    
    sinal_avaliado = sinal_avaliado.reshape(-1,1)
    sinal_avaliado = sinal_avaliado[~np.isnan(sinal_avaliado).any(axis=1)] # Eliminação de linha com nan
    
    plt.plot(sinal_avaliado)
    plt.grid()
    plt.show()
    
    # Passa potência em dBm para volts
    # --------------------------------
    sinal_em_dbm = sinal_avaliado
    test_array = np.sqrt(1e-3*10**(sinal_em_dbm/20))
    
    test_array = sinal_avaliado
    
    bins = 25
    
    sse, best_name, best_params, x_mid, \
    pdf_scaled, new_x, best_pdf_scaled, best_pdf, x, y \
    = fit_pdf.fit_scipy_distributions(test_array, bins,
                                      plot_best_fit=True,
                                      plot_all_fits=True)
    # Gráfico
    '--------'
    plt.title('Histogram and pdf - ' + conta_med)
    plt.xlabel('data')     # 'Distância - km'
    plt.ylabel('Density')    # 'Nível do sinal - dB'
    
    plt.hist(np.array(test_array), bins = bins, 
             facecolor='none', 
             edgecolor='blue', 
             label='Histogram',
             histtype = 'bar',
             alpha = 0.5,
             density = True
             )
    plt.plot(new_x, best_pdf, color = 'r', label = best_name)
    
    plt.legend(loc='best',  fontsize = 10)
    
    plt.grid()
    plt.show()
    #
    # Formação da tabela para avaliação
    '----------------------------------'
    
    if controle_tabela == 0:
        
        sse_tabela_df = sse
        
        sse_tabela_df['melhor_pdf'] = best_name
        sse_tabela_df['Rota'] = conta_med
        
    else:
        sse_tab_df = sse
        
        sse_tab_df['melhor_pdf'] = best_name
        sse_tab_df['Rota'] = conta_med
        
        sse_tabela_df = pd.concat([sse_tabela_df, sse_tab_df])
        
    '----------------------------------'
    controle_tabela += 1
    
    ''' ---------------------------------------------------- '''
    # ENTER = input('\nPressione [ENTER] para continuar...\n')
    ''' ---------------------------------------------------- '''
    #
#
# Salva tabela de resultados no arquivo excel - sinal escolhido
'--------------------------------------------------------------'
#
''' Ref.:
https://stackoverflow.com/questions/42370977/how-to-save-a-new-sheet-in-an-existing-excel-file-using-pandas
'''
# os.getcwd()
caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/Avaliacao_pdf/"
nome_arq_xlsx = ''.join(var_entrada) + '_pdf.xlsx'
pasta_excel = ''.join(var_entrada)

sse_tabela_df.to_excel(caminho + nome_arq_xlsx,
                       sheet_name = pasta_excel, index = True)
#
# %% pdf - Teste de verificação de pdf de melhor ajuste - dsitfit
' --------------------------------------------------------------- '
#
'''
Ref:
https://towardsdatascience.com/how-to-find-the-best-theoretical-distribution-for-your-data-a26e5673b4bd?gi=fc805b018db2
https://erdogant.github.io/distfit/pages/html/index.html
https://stackoverflow.com/questions/7125009/how-to-change-legend-fontsize-with-matplotlib-pyplot
https://stackoverflow.com/questions/34001751/how-to-increase-reduce-the-fontsize-of-x-and-y-tick-labels
https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
https://stackoverflow.com/questions/24039023/add-column-with-constant-value-to-pandas-dataframe
'''
# Loop para avaliação de todas as rotas
' ------------------------------------- '
#
'--------------------------------'
teste_medicao = medicoes_rota
# Comentar a linha abaixo se não for utilizar
teste_medicao = ['R1H1F4'] # Se especificado executa somente o indicado
'--------------------------------'
#
contr_conta_loop = 0
for conta_med in teste_medicao: # medicoes_rota:
    #
    # Escolha das vaiáveis de entrada e saída
    ' ----------------------------------------- '
    #
    ' -------- Estudo 1 -------- '
    var_entrada = ['altura_tx', 'freq_tx',
                   'dist_rota_desv_len_somb_pathloss_mseg'] #, 'class_autom_2']
    var_saida = ['potRecNP_desv_len_somb_pathloss_mseg'] #, 'class_autom_2']

    var_x = ['dist_rota_vab_rap_mseg']
    var_y = ['potRecNP_var_rap_mseg']

    var_comparacao = ['Prec_ModLogNorm']
    var_comparacao_dist = ['dist_rota']

    ' ----- sinal ------ '
    sinal_avaliado =  locals()[conta_med][var_y].values
    sinal_avaliado = sinal_avaliado.reshape(-1,1)
    sinal_avaliado = sinal_avaliado[~np.isnan(sinal_avaliado).any(axis=1)] # Eliminação de linha com nan

    ' ----- distância ------ '
    sinal_aval_dist =  locals()[conta_med][var_x].values
    sinal_aval_dist = sinal_aval_dist.reshape(-1,1)
    sinal_aval_dist = sinal_aval_dist[~np.isnan(sinal_aval_dist).any(axis=1)] # Eliminação de linha com nan

    ' ------ Gráfico de teste ---- '
    plt.plot(sinal_aval_dist, sinal_avaliado)
    plt.grid()
    plt.show()

    # Segmentação do sinal
    ' -------------------- '
    #
    var_1 = 'quant_seg_mm'
    valor_var_1 =  int(locals()[conta_med][var_1].dropna().values)
    
    comp_var_ent = int(len(sinal_avaliado)/valor_var_1)
    # comp_var_ent_final = 
    
    # Inicialização com configuração paramétrica
    '-------------------------------------------'
    distribuicoes = ['norm', 'expon', 'nakagami', 'rice', 'rayleigh', 'lognorm']
    dfit = distfit(method= 'parametric', todf=True, distr = distribuicoes, verbose=0)
    
    # Parâmtros do gráfico
    '---------------------'
    tam_texto = 35
    param_global_grafico = {'legend.fontsize': 'x-large',
         # 'figure.figsize': (15, 5),
         'axes.labelsize': tam_texto,
         'axes.titlesize': tam_texto,
         'xtick.labelsize': tam_texto,
         'ytick.labelsize': tam_texto,
         'legend.fontsize': tam_texto}
    pylab.rcParams.update(param_global_grafico)
        
    # # Inicialização com configuração paramétrica utilizando bootstrapping -> muito lento
    # dfit = distfit(method= 'parametric', todf=True, n_boots=100)
    
    # data frame com o resultado dos cálculos
    resultado_total_df = pd.DataFrame
    
    valor_ini = 0
    valor_inc = comp_var_ent
    valor_final = valor_ini + valor_inc
    
    for conta_valor in range(valor_var_1):
        #
        seg_sinal_aval = sinal_avaliado[valor_ini:valor_final]
        seg_sinal_aval_dsit = sinal_aval_dist[valor_ini:valor_final]
        
        valor_ini = valor_final
        valor_final = valor_ini + valor_inc
        
        # Gráfico do sinal
        plt.plot(seg_sinal_aval_dsit, seg_sinal_aval)
        plt.show()
    
        # Calcula pdf
        resultado = dfit.fit_transform(seg_sinal_aval)
        print(dfit.summary[['name', 'score', 'loc', 'scale']])
        
        # Cria um data frame
        result_df = dfit.summary
        
        # Acrescenta coluna com a posição do segmento
        result_df['Segmento_indice'] = conta_valor
        
        # Gráficos
        ' -------- '
        # Pareto plot (PDF with histogram)
        fig, ax = dfit.plot(chart='PDF', n_top = 2,
                            xlabel = 'Data \n',
                            ylabel = 'Density \n',
                            verbose = None,
                            pdf_properties={'color': 'r', 'linewidth': 4, 'linestyle': 'dashed', 'marker': 'x'},
                            emp_properties = None,
                            cii_properties = None,
                            bar_properties={'color': 'white', 'linewidth': 3,
                                            'edgecolor': 'blue', 'align': 'edge'})
        ax.set_title('Histogram and best fit - ' + conta_med + ' Seg: ' + str(conta_valor) + '\n')
        ax.legend()
        plt.show()
        
        nome_melhor_modelo = dfit.model['name']
        print(nome_melhor_modelo)
    
        # freq, bins = dfit.histdata
        # plt.plot(bins, freq)
        
        # # Change or remove properties of the chart.
        # dfit.plot(chart='PDF', 
        #       emp_properties=None,
        #       bar_properties=None,
        #       pdf_properties={'color': 'r'},
        #       cii_properties={'color': 'g'})
    
        # Plot the CDF
        # fig, ax = dfit.plot(chart='CDF', n_top = 1)
        
        plt.show()
        
        # Plot the top fitted distributions.
        grafico = dfit.plot_summary()
        #
        # Tabela 1 - planilha 2 - TABELA / tabela
        # Formação da tabela para avaliação do segmento
        ' --------------------------------------------- '
        if conta_valor == 0:
            
            # Nome das colunas
            tabela_nomes_dist = distribuicoes
            
            # Cria DataFrame vazio
            tabela_dados_df = pd.DataFrame(columns = tabela_nomes_dist)
                    
            # Adiciona valor na linha da coluna
            tabela_dados_df.loc[len(tabela_dados_df.index), [nome_melhor_modelo]] = nome_melhor_modelo
            
            # Adiciona coluna da melhor pdf
            tabela_dados_df['Melhor pdf'] = nome_melhor_modelo
            
            # Formação da tabela da rota
            tabela_dados_rota_df = tabela_dados_df
            
        else:
            # Adiciona valor na linha da coluna
            tabela_dados_df.loc[len(tabela_dados_df.index), [nome_melhor_modelo]] = nome_melhor_modelo

            # Adiciona coluna da melhor pdf
            tabela_dados_df.loc[len(tabela_dados_df.index)-1, ['Melhor pdf']] = nome_melhor_modelo
            
            # Formação da tabela da rota
            tabela_dados_rota_df = pd.concat([tabela_dados_rota_df, tabela_dados_df])
            
            #
        print(tabela_dados_df)
        #
        # Tabela 2 - planilha 1 - RESULTADO / resultado
        # Formação da tabela para avaliação do rota (todos os segmentos)
        ' -------------------------------------------------------------- '
        if conta_valor == 0:
            #
            # Cabeçalho das colunas
            cab_resultado = result_df.columns.tolist()
            # Data frame vazio
            resultado_total_df = pd.DataFrame(columns = cab_resultado)
            # Concatena dadso do datraframe de resultado
            resultado_total_df = pd.concat([resultado_total_df, result_df])
        
        else:
            # Concatena dadso do datraframe de resultado
            resultado_total_df = pd.concat([resultado_total_df, result_df])
        #
        # Bootstrapping -> muito lento
        # print(dfit.summary[['name', 'score', 'loc', 'scale', 
        #                     'bootstrap_score', 'bootstrap_pass']])
        
        # Parada para avaliação
        '-------------------------------------------------------'
        # ENTER = input('\nPressione [ENTER] para continuar...\n')
        '-------------------------------------------------------'
    #
    # Salva tabela de resultados no arquivo excel - por rota
    ' ------------------------------------------------------ '
    #
    ''' Ref.:
    https://stackoverflow.com/questions/42370977/how-to-save-a-new-sheet-in-an-existing-excel-file-using-pandas
    '''
    # os.getcwd()
    data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")
    #
    caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/Avaliacao_pdf/"
    nome_arq_xlsx = conta_med + '_pdf_' + data_hora_gravacao + '.xlsx'
    pasta_excel = conta_med
    resultado_total_df.to_excel(caminho + nome_arq_xlsx,
                                sheet_name = pasta_excel, index = True)
    
    # Acrescenta nova pasta no excel
    with pd.ExcelWriter(caminho + nome_arq_xlsx, engine='openpyxl', mode='a') as writer:  
        tabela_dados_df.to_excel(writer, sheet_name='Resultado_seg')
    #
    # Formação de uma planilha geral (TABELA + RESULTADO) para posterior análise
    ' -------------------------------------------------------------------------- '
    if contr_conta_loop == 0:
        #
        # Acrescenta coluna com a rota de estudo
        resultado_total_df['Rota'] = conta_med
        resultado_total_rotas_df = resultado_total_df
        
        tabela_dados_rota_df['Rota'] = conta_med
        tabela_dados_total_rota_df = tabela_dados_rota_df
        
    else:
        # Acrescenta coluna com a rota de estudo
        resultado_total_df['Rota'] = conta_med
        # Concatena
        resultado_total_rotas_df = pd.concat([resultado_total_rotas_df,
                                              resultado_total_df])
        #
        tabela_dados_rota_df['Rota'] = conta_med
        # Concatena
        tabela_dados_total_rota_df = pd.concat([tabela_dados_total_rota_df,
                                                tabela_dados_rota_df])
        #
    contr_conta_loop += 1
    ' ------------------- '
    #
    '-------------------------------------------------------'
    # ENTER = input('\nPressione [ENTER] para continuar...\n')
    '-------------------------------------------------------'
#
# %% Salva tabela de resultados no arquivo excel - todas as rotas
' ------------------------------------------------------------ '
#
''' Ref.:
https://stackoverflow.com/questions/42370977/how-to-save-a-new-sheet-in-an-existing-excel-file-using-pandas
'''
# os.getcwd()
data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")
#
caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/Avaliacao_pdf/"
nome_arq_xlsx = 'TodasAsRotas_pdf_' + data_hora_gravacao + '.xlsx'
pasta_excel = 'TodasAsRotas'

resultado_total_rotas_df.to_excel(caminho + nome_arq_xlsx,
                                  sheet_name = pasta_excel, index = True)
#
# Acrescenta nova pasta no excel
with pd.ExcelWriter(caminho + nome_arq_xlsx, engine='openpyxl', mode='a') as writer:  
    tabela_dados_total_rota_df.to_excel(writer, sheet_name='ResTadasRotasSeg')

#     
#%% Cálculo dos erros em relação aos modelos - dBm
'------------------------------------------------------------------------------'
#
MSE_EspacoLivre = mean_squared_error(potRec, Prec_AtenEspLivre) # Prec_AtenEspLivre
MSE_TerraPlana = mean_squared_error(potRec, Prec_ModTerraPlan)
MSE_LogNormal = mean_squared_error(potRec, Prec_ModLogNorm)

RMSE_LogNormal = np.sqrt(MSE_LogNormal)

print('RMSE Free Space:', np.sqrt(MSE_EspacoLivre))
print('RMSE Flat Earth:', np.sqrt(MSE_TerraPlana))
print('RMSE Log-Normal:', np.sqrt(MSE_LogNormal))

print('MSE Free Space:', MSE_EspacoLivre)
print('MSE Flat Earth:', MSE_TerraPlana)
print('MSE Log-Normal:', MSE_LogNormal)

r2_EspacoLivre = r2_score(potRec, Prec_AtenEspLivre) # Prec_AtenEspLivre
r2_TerraPlana = r2_score(potRec, Prec_ModTerraPlan)
r2_LogNormal = r2_score(potRec, Prec_ModLogNorm)

print('R^2 Free Space:', r2_EspacoLivre)
print('R^2 Flat Earth:', r2_TerraPlana)
print('R^2 Log-Normal:', r2_LogNormal)
#
#%% Gráficos - Desenho dos gráficos escolhidos
'------------------------------------------------------------------------------'
#
# for conta_med, conta_nome in zip(nome_dados_rota, medicoes_rota):
for conta_med in medicoes_rota:
    #
    medicao = conta_med # + '_df_proc'
    # print(conta_med, conta_nome)
    #
    # -------------------------------------------------------------------------
    escolha = ''
    while not escolha:
        #
        #
        # Cabeçalho
        # ---------
        escolha_col = ''
        conta_pos_col = 0
        for element in cab_colunas_rota_01:
            print(str(conta_pos_col) + ' - ' + element)
            conta_pos_col +=1
        #
        # Escolha das variáveis de entrada
        # --------------------------------
        var_x = []
        while True:
            lista = input('\n Entre número da posição da variável de entrada [X]: ');
            if len(lista) == 0:
                break
            var_x.append(cab_colunas_rota_01[int(lista)])
        #
        print('\n---------------------------------------------------------------------')
        print('Variáveis de entrada: ')
        print(var_x)
        print('---------------------------------------------------------------------')
        #
        # Escolha das variáveis de saída
        # ------------------------------
        var_y = []
        while True:
            lista = input('\n Entre número da posição da variável de saída:[Y] ');
            if len(lista) == 0:
                break
            var_y.append(cab_colunas_rota_01[int(lista)])
        #
        print('\n---------------------------------------------------------------------')
        print('Variável(is) de saída(s): ')
        print(var_y)
        print('---------------------------------------------------------------------')
        #
        #
        #
        num_conta = 0
        for conta in var_x:
            #
            if num_conta == 0:
                valor = locals()[conta_med][conta].values
                conta
                var_x_df = pd.DataFrame(valor, index = None, columns=[conta])
                num_conta +=1
            else:
                valor = locals()[conta_med][conta].values
                conta
                var_df_temp = pd.DataFrame(valor, index = None, columns=[conta])

                var_x_df = pd.concat([var_x_df, var_df_temp], axis = 1)
                num_conta +=1
        var_x_array = var_x_df.to_numpy()
        
        num_conta = 0
        for conta in var_y:
            #
            if num_conta == 0:
                valor = locals()[conta_med][conta].values
                conta
                var_y_df = pd.DataFrame(valor, index = None, columns=[conta])
                num_conta +=1
            else:
                valor = locals()[conta_med][conta].values
                conta
                var_df_temp = pd.DataFrame(valor, index = None, columns=[conta])

                var_y_df = pd.concat([var_y_df, var_df_temp], axis = 1)
                num_conta +=1
        var_y_array = var_y_df.to_numpy()

        # Gráfico
        # -----------------------
        #
        plt.title('Signal attenuation')
        plt.xlabel('Distance - m')         # 'Distância - km'
        plt.ylabel('Signal received - dBm')    # 'Nível do sinal - dB'
        
        cores = ['b', 'r', 'g']
        for conta_plot in np.arange(len(var_x)):
            plt.plot(var_x_array[:,conta_plot], var_y_array[:,conta_plot],
                     label = cores[conta_plot], color = cores[conta_plot] )
        
        # plt.text(dist_pot_rec[-1], aten_pot_rec[-1],
        #          conta_med, c = 'b', fontsize  = 12)
        plt.legend(loc='best',
                    fontsize = 8)
        
        #% Escolha das vaiáveis para comparação - Log-distância
        ''' --------------------------------------------------- '''
        #
        var_comparacao = 'Prec_ModLogNorm'
        var_comparacao_dist = 'dist_rota'

        #% Comparação com referência -> log-distância
        ''' ----------------------------------------------------------------------- '''
        #
        # Valores individuais:
        ''' ------------------ '''
        n = locals()[medicao]['n'].values[0]
        p_c = locals()[medicao]['p_c'].values[0]

        n_aten_sinal = locals()[medicao]['n_aten_sinal'].values[0]
        p_c_aten_sinal = locals()[medicao]['p_c_aten_sinal'].values[0]
        
        n_aval_ref = n
        p_c_aval_ref = p_c
        
        # Valores processados e para comparação:
        ''' ------------------------------------ '''
        log_distance = locals()[medicao][var_comparacao].values
        log_distance = log_distance.reshape(-1,1)
        log_distance = log_distance[~np.isnan(log_distance).any(axis=1)] # Eliminação de linha com nan
        #        
        dist_log_distance = locals()[medicao][var_comparacao_dist].values
        dist_log_distance = dist_log_distance.reshape(-1,1)
        dist_log_distance = dist_log_distance[~np.isnan(dist_log_distance).any(axis=1)] # Eliminação de linha com nan

        dist_p_r = dist_log_distance # X_predicao[:,-1]
        p_r = p_c_aval_ref + 10 * np.log10(dist_p_r/dist_p_r[0]) * n_aval_ref
        #        #
        # Gráfico - log-distância
        # -----------------------
        # plt.plot(dist_pot_rec, aten_pot_rec, c = 'b', lw = 0.5, label = conta_med)
        plt.plot(dist_p_r, p_r, c = 'r', lw = 1, label = 'Log-distância')

        # -----------------------------
        plt.grid()
        plt.show()
        #
        #
        #
        ''' ---------------------------------------------------------------- '''
        escolha = input('\n nova escolha [s]/n ?')
        if len(escolha) == 0 or escolha == 's':
            escolha = ''
        else:
            escolha = 'n'
        ''' ---------------------------------------------------------------- '''
#
#
#%% Gráficos de avaliação dos modelos de atenuação
'------------------------------------------------------------------------------'
#
'''
Ref.:
https://matplotlib.org/2.0.0/mpl_examples/color/named_colors.png
https://github.com/spyder-ide/spyder/issues/8463'
'''
#
import SinalVegetacao_modelosVegetacao_v0 as mod_veg
#
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# Alguns parâmetros gerais dos gráficos
'--------------------------------------'
largura_linha = 1.25
# fig.set_size_inches(20.5, 5.5)
# fig_width, fig_height = 6.4, 4.8
plt.rcParams["figure.figsize"] = (7, 5)
# plt.axis([37, 73, 70, 140,])
#
'--------------------------------'
teste_medicao = medicoes_rota
# Comentar a linha abaixo se não for utilizar
teste_medicao = ['R1H2F4'] # Se especificado executa somente o indicado
'--------------------------------'
#
val_conta_rota_proc = 0
val_conta_med = 1
for conta_med in teste_medicao:
    
    # Atenuação
    # ---------
    EIRP = locals()['EIRP' + str(val_conta_med)]
    Gt = locals()['G' + str(val_conta_med) +'_r' ]
    Gr = locals()['G' + str(val_conta_med) +'_t' ]
    AmpRec = 0
    P_cabos = 2
    Ptrans = 40
    frequencia = locals()[conta_med]['freq_tx'][0]
    
    pot_rec = locals()[conta_med]['Pr']
    pot_rec = locals()[conta_med]['potRecNP_pathloss']
    pot_rec = pot_rec[~np.isnan(pot_rec)]
    '-----------------------------------------------------'
    Atenua_sinal_pot_rec = mod_veg.AtenuaSinalPotRec(EIRP, Ptrans, Gt, Gr, AmpRec,
                                             P_cabos,
                                             pot_rec)
    
    Atenua_sinal_pot_rec = locals()[conta_med]['Atenua_sinal_pot_rec_mseg']
    '-----------------------------------------------------'
    
    distancia = locals()[conta_med]['d_radial']
    distancia = locals()[conta_med]['dist_rota_pathloss']
    
    distancia = distancia[~np.isnan(distancia)]

    '---------------------------------------------------'
    comp_var_dist = len(distancia)
    comp_var_aten = len(Atenua_sinal_pot_rec)
    if comp_var_dist <= comp_var_aten:
        Atenua_sinal_pot_rec = Atenua_sinal_pot_rec[0:comp_var_dist]
    else:
        distancia = distancia[0:comp_var_aten]
    '---------------------------------------------------'


    # Atenuação pelo sinal medido
    '----------------------------'
    plt.plot(distancia, Atenua_sinal_pot_rec,
             c = 'b', ls = '--', lw = largura_linha,
             marker='.',
             markersize=10,
             markevery= 200,
             label = 'Atenuação sinal medido')

    # Log-distância
    '--------------'
    aten_mod_log_distance = locals()[conta_med]['aten_mod_log_distance']
    distancia_log_dist = locals()[conta_med]['dist_rota']
    
    plt.plot(distancia_log_dist, aten_mod_log_distance,
              c = 'm', lw = (largura_linha),
              marker='x',
              markersize=7,
              markevery= 400,
              label = 'Log-distância')
    #
    #
    #
    #% Comparação com referência -> log-distância
    '--------------------------------------------'    
    #
    # Valores individuais:
    '---------------------'
    n = locals()[conta_med]['n'].values[0]
    p_c = locals()[conta_med]['p_c'].values[0]

    n_aten_sinal = locals()[conta_med]['n_aten_sinal'].values[0]
    p_c_aten_sinal = locals()[conta_med]['p_c_aten_sinal'].values[0]
    
    n_aval_ref = n_aten_sinal
    p_c_aval_ref = p_c_aten_sinal
    
    dist_p_r = distancia
    p_r = p_c_aval_ref + 10 * np.log10(dist_p_r/dist_p_r[0]) * n_aval_ref
    #
    # Gráfico - log-distância
    # -----------------------
    # plt.plot(dist_pot_rec, aten_pot_rec, c = 'b', lw = 0.5, label = conta_med)
    # plt.plot(dist_p_r, p_r,
    #          c = 'r', lw = largura_linha, label = 'Log-distância')
    #
    #
    #
    # Atenuação espaço livre
    # ----------------------
    A_esp_livre = mod_veg.perdEspLivre(frequencia/1e6, distancia/1000)
    plt.plot(distancia, A_esp_livre,
             c = 'g', lw = largura_linha,
              marker= None ,
              markersize=10,
              markevery= None,
              label = 'Espaço livre')

    # Modelo ITU-R / ITU Early ITU
    # ----------------------------
    A_itu_r = mod_veg.modelo_itu_r(frequencia/1e6, distancia)
    A_itu_r = A_itu_r + A_esp_livre
    
    plt.plot(distancia, A_itu_r,
              c = 'c',lw = largura_linha,
              marker='o',
              markersize=5,
              markevery= 250,
              label = 'Early ITU')
    # 
    # Modelo FITU-R
    # -------------
    A_fitu_r_com_folhas, A_fitu_r_sem_folhas = mod_veg.modelo_fitu_r(frequencia/1e6,
                                                                     distancia)
    A_fitu_r_com_folhas = A_fitu_r_com_folhas + A_esp_livre
    A_fitu_r_sem_folhas = A_fitu_r_sem_folhas + A_esp_livre
    
    plt.plot(distancia, A_fitu_r_com_folhas,
              c = 'k', lw = largura_linha,
              marker='1',
              markersize=10,
              markevery= 250,
              label = 'FITU-R com folhas')
    # plt.plot(distancia, A_fitu_r_sem_folhas,
    #          c = 'k', lw = largura_linha, label = 'FITU-R sem folhas')

    # Modelos Weissberger
    # -------------------
    A_weiss_ate_14m, A_weiss_14_400m = mod_veg.modelo_weissberger(frequencia/1e9,
                                                                  distancia)
    A_weiss_ate_14m = A_weiss_ate_14m + A_esp_livre
    A_weiss_14_400m = A_weiss_14_400m + A_esp_livre
    
    # plt.plot(distancia, A_weiss_ate_14m, c = 'y',
    #                   lw = largura_linha, label = 'Weissberger d < 14m')
    plt.plot(distancia, A_weiss_14_400m, c ='r', ls = '-',
                      lw = largura_linha,
                      marker='s',
                      markersize=7,
                      markevery= 400,
                      label = 'Weissberger 14m < d < 400m')

    # Modelo Chen and Kuo
    # -------------------
    A_chen_kuo_vert, A_chen_kuo_hor = mod_veg.modelo_chen_kuo(frequencia/1e9,
                                                              distancia)
    A_chen_kuo_vert = A_chen_kuo_vert + A_esp_livre
    A_chen_kuo_hor = A_chen_kuo_hor + A_esp_livre
    
    plt.plot(distancia, A_chen_kuo_vert, c = 'gold',
                      marker='x',
                      markersize=10,
                      markevery= 500,
                      lw = largura_linha, label = 'Chen and Kuo - vert')
    # plt.plot(distancia, A_chen_kuo_hor,
    #                   lw = largura_linha, label = 'Chen and Kuo - horiz')
    
    
    plt.title('Atenuação do sinal - ' + conta_med )
    plt.xlabel('Distância (m)')         # 'Distância - km'
    plt.ylabel('Atenuação (dB)')    # 'Nível do sinal - dB'
    
    plt.legend(loc='best',fontsize = 8)
    
    # plt.axis((35,80,70, 140))
    
    x1,x2,y1,y2 = plt.axis() 
    plt.axis((38,72,70,140))
    plt.grid()
    plt.show()
    
    # Cálculos dos erros RMSE
    '------------------------'
    RMSE_Espaco_livre = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_esp_livre))
    RMSE_Log_distancia = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, p_r))
    
    RMSE_A_itu_r = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_itu_r ))
    
    RMSE_A_fitu_r_com_folhas = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_fitu_r_com_folhas ))
    RMSE_A_fitu_r_sem_folhas = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_fitu_r_sem_folhas))
    
    RMSE_A_weiss_ate_14m = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_weiss_ate_14m))
    RMSE_A_weiss_14_400m = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_weiss_14_400m))

    RMSE_A_chen_kuo_vert = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_chen_kuo_vert ))
    RMSE_A_chen_kuo_hor = np.sqrt(mean_squared_error(Atenua_sinal_pot_rec, A_chen_kuo_hor))

    # Formação de tabelas para análise
    '---------------------------------'
    if val_conta_rota_proc == 0:
        tabele_nomes = ['Rota de medição', 'RMSE_Espaço_livre', 'RMSE_Log_distancia', 'RMSE_A_itu_r',
                        'RMSE_A_fitu_r_com_folhas', 'RMSE_A_fitu_r_sem_folhas',
                        'RMSE_A_weiss_ate_14m', 'RMSE_A_weiss_14_400m',
                        'RMSE_A_chen_kuo_vert', 'RMSE_A_chen_kuo_hor']
        
        tabela_dados = [conta_med ,RMSE_Espaco_livre, RMSE_Log_distancia, RMSE_A_itu_r,
                        RMSE_A_fitu_r_com_folhas, RMSE_A_fitu_r_sem_folhas,
                        RMSE_A_weiss_ate_14m, RMSE_A_weiss_14_400m,
                        RMSE_A_chen_kuo_vert, RMSE_A_chen_kuo_hor]
        
        # Cria DataFrame vazio
        tabela_dados_df = pd.DataFrame(columns = tabele_nomes)
        # Adiciona linhas
        tabela_dados_df.loc[len(tabela_dados_df)] = tabela_dados
        
    else:
        
        tabela_dados = [conta_med, RMSE_Espaco_livre, RMSE_Log_distancia, RMSE_A_itu_r,
                        RMSE_A_fitu_r_com_folhas, RMSE_A_fitu_r_sem_folhas,
                        RMSE_A_weiss_ate_14m, RMSE_A_weiss_14_400m,
                        RMSE_A_chen_kuo_vert, RMSE_A_chen_kuo_hor]
        # Adiciona linhas
        tabela_dados_df.loc[len(tabela_dados_df)] = tabela_dados
    #    
    val_conta_rota_proc += 1
    #
    # ''' ---------------------------------------------------- '''
    # ENTER = input('\nPressione [ENTER] para continuar...\n')
    # ''' ---------------------------------------------------- '''
#
#%% Gravação do arquivo tabela de dados
'------------------------------------------------------------------------------'
#
data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")

caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/"

pasta_excel = 'rota_' + nome_rota_proc
nome_arq_excel = 'tabelaComparaModelosVegetacao_rota_' + nome_rota_proc + '_path_loss'
nome_arq_xlsx = nome_arq_excel + '-' + data_hora_gravacao + '.xlsx'

# Gravação do arquivo
''' ----------------- '''
tabela_dados_df.to_excel(caminho + nome_arq_xlsx,
                                sheet_name = pasta_excel,
                                index = False)
#
''' Marcas importantes
# --------------------'''
#
# pontInfVert = -70
# pontSupVert = -25

# pontoMarca = 1450 / 1000
# # plt.vlines(pontoMarca, pontSupVert,pontInfVert, colors = 'k', linestyles = 'dashed', linewidth=0.75)
# plt.vlines(518.36 / 1000, pontInfVert, pontSupVert, colors = 'r', linestyles = 'dashed', linewidth=0.75)
# plt.vlines(781.71 / 1000, pontInfVert, pontSupVert, colors = 'r', linestyles = 'dashed', linewidth=0.75)
# plt.vlines(1339.65 / 1000, pontInfVert, pontSupVert, colors = 'b', linestyles = 'dashed', linewidth=0.75)

# posEixoX = 0.55
# posEixoX_01 = 0.00
# posEixoX_02 = 0.80
# posEixoX_03 = 1.35

# posEixoY = -40
# posEixoY_01 = -60
# posEixoY_02 = -70
# posEixoY_03 = -82

# plt.annotate('snippet\n with \n missing \n data',
#              xy=(posEixoX,posEixoY), rotation=0, fontsize=8, color = 'red')
# plt.annotate('segment 01',
#              xy=(posEixoX_01,posEixoY_01), fontsize=12, color = 'blue')
# plt.annotate('segment 02',
#              xy=(posEixoX_02,posEixoY_02), fontsize=12, color = 'black')
# plt.annotate('segment 03',
#              xy=(posEixoX_03,posEixoY_03), fontsize=12, color = 'green')
#