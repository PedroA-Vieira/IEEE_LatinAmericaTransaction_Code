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
Descrição:
'''
#
#%% Bibliotecas
'------------------------------------------------------------------------------'
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import time
import os

from scipy import signal

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
                                    cross_val_score, \
                                    cross_validate, \
                                    StratifiedKFold, \
                                    LeaveOneOut
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
# Atenuação da potência do sinal pecebido: EIRP, Ptrans, Gt, Gr, Glna, Acc, Ao
def AtenuaSinalPotRec(EIRP=0, Ptrans=0, Gt=0, Gr=0, Glna=0, Acc=0, potRec=0):
    '''
    Calcula a atenuação do sinal á partir da potência recebida medida'''
    aten_sinal_pot_rec = EIRP + (Ptrans + Gt) + Gr + Glna - Acc - potRec
    return aten_sinal_pot_rec
#
try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='inline') # 'inline') qt5
except:
    pass
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

# comp_nome_3 = '_df_proc_saida'
# comp_nome_4 = '_df_proc_3'
#
# %% Leitura dos arquivos excel
'------------------------------------------------------------------------------'
#
'x=--> Pode passar para leitura do dataframe '

print("\014") # Clear console
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
# %% Armazenamento do dataframe
'------------------------------------------------------------------------------'
#
caminho_salva_df_1 = r'Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao'
caminho_salva_df_2 = '\Artigo_03_Vegetacao\MedicoesProcessados\ArquivosDataFrame'
caminho_salva_df = caminho_salva_df_1 + caminho_salva_df_2
#
for conta_med in medicoes_rota_01:
    locals()[conta_med].to_pickle(caminho_salva_df + '/' + conta_med + '.pkl')
#
for conta_med in medicoes_rota_02:
    locals()[conta_med].to_pickle(caminho_salva_df + '/' + conta_med + '.pkl')
#
# %% Leitura do dataframe
'------------------------------------------------------------------------------'
#
for conta_med in medicoes_rota_01:
    print('-->> Lendo ' + conta_med)
    locals()[conta_med] = pd.read_pickle(caminho_salva_df + '/' + conta_med + '.pkl')
#
for conta_med in medicoes_rota_02:
    print('-->> Lendo ' + conta_med)
    locals()[conta_med] = pd.read_pickle(caminho_salva_df + '/' + conta_med + '.pkl')
#
#%% Obtem lista com o cabeçcalho das colunas
'------------------------------------------------------------------------------'
#
' --- Cabeçalhos colunas: --- '
' ----------------------------'
cab_colunas = locals()[medicoes_rota_01[0]].columns.tolist()
#
medicoes_rota = medicoes_rota_01 + medicoes_rota_02
#
cab_colunas_rota_01 = locals()[medicoes_rota_01[0]].columns.tolist()
cab_colunas_rota_02 = locals()[medicoes_rota_02[0]].columns.tolist()
#
# %% Escolha da rota de processamento
'------------------------------------------------------------------------------'
#
print("\014") # Clear console
#
controla_escolha = 0
controla_escolha_rede_treina = 0
while controla_escolha_rede_treina==0:
    #
    nome_rota_proc = input('\n\nQual a rota para processamento? [1]/2/3=->ambas:')
    if len(nome_rota_proc)==0 or nome_rota_proc == '1':
        nome_rota_proc = '1'
        medicoes_rota = medicoes_rota_01
        # cab_colunas = cab_colunas_rota_01
        controla_escolha_rede_treina = 1
        
    elif nome_rota_proc == '2':
        medicoes_rota = medicoes_rota_02
        # cab_colunas = cab_colunas_rota_02
        controla_escolha_rede_treina = 2
        
    elif nome_rota_proc == '3':
        medicoes_rota = medicoes_rota_01 + medicoes_rota_02
        # cab_colunas = cab_colunas_rota_02
        controla_escolha_rede_treina = 3 
        
    else:
        nome_rota_proc = ''
        controla_escolha_rede_treina = 0
#
# %%
''' ---------------------------------------------------- '''
ENTER = input('\nPressione [ENTER] para continuar...\n')
''' ---------------------------------------------------- '''
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
#%% Escolha dos dados para treinamento da rede neural
'------------------------------------------------------------------------------'
#
# As rotas devem ser escolhidas aqui, valendo a última desmarcada
''' -------------------------------------------------------------- '''
#
# medicoes_estudo_df =  pd.concat([R1H1F1_df_proc, R1H1F2_df_proc, R1H1F3_df_proc,
#                           R1H1F4_df_proc, R1H1F5_df_proc,
#                           R1H2F1_df_proc, R1H2F2_df_proc, R1H2F3_df_proc,
#                           R1H2F4_df_proc, R1H2F5_df_proc]).sort_index().reset_index(drop=True)
#
# medicoes_estudo_df = pd.concat([R2H1F1_df_proc, R2H1F2_df_proc, R2H1F3_df_proc,
#                           R2H1F4_df_proc, R2H1F5_df_proc,
#                           R2H2F1_df_proc, R2H2F2_df_proc, R2H2F3_df_proc,
#                           R2H2F4_df_proc, R2H2F5_df_proc]).sort_index().reset_index(drop=True)
#
# medicoes_estudo_df =  pd.concat([R1H1F1_df_proc, R1H1F5_df_proc]).sort_index().reset_index(drop=True)
#
# medicoes_estudo_df =  pd.concat([R1H1F1_df_proc]).sort_index().reset_index(drop=True)
#
# Formação dos dados de estudo
''' -------------------------- '''
#
# Lista das rotas de estudo
# -------------------------
conta_pos_col = 0
for conta_med in medicoes_rota:
    print(str(conta_pos_col) + ' - ' + conta_med)
    conta_pos_col +=1
#
# Escolha das rotas de estudo
# ---------------------------
var_medicoes_estudo = []
while True:
    lista = input('\nEntre número da(s)) rota(s) de estudo:\n' \
                  '[ENTER] =-> encerra, T / t =-> tudo ');
    if lista.lower() == 't':
        var_medicoes_estudo = medicoes_rota
        break
    if len(lista) == 0:
        break
    var_medicoes_estudo.append(medicoes_rota[int(lista)])
#
medicoes_estudo_df = pd.DataFrame()     # Cria dataframe vazio
for conta_med in var_medicoes_estudo:
    medicoes_estudo_df = pd.concat([medicoes_estudo_df, locals()[conta_med]])
    print('\n Adicionado rota --> '+ conta_med)
#
# medicoes_estudo_df = pd.DataFrame()
# for conta_med in medicoes_rota:
#     medicoes_estudo_df = pd.concat([medicoes_estudo_df, locals()[conta_med]])
#
print('\n---------------------------------------------------------------------')
print('Variáveis de entrada: ')
print(var_medicoes_estudo)
print('---------------------------------------------------------------------')
#
#%% Escolha das vaiáveis de entrada e saída
'------------------------------------------------------------------------------'
#
print("\014") # Clear console
#
# Variáveis padrão
# ----------------
# var_entrada = ['altura_tx', 'freq_tx', 'd_radial', 'Prec_ModLogNorm']
# var_entrada = ['altura_tx', 'freq_tx', 'd_radial', 'Ptrans']
# var_entrada = ['altura_tx', 'freq_tx', 'd_radial',
#                'class_autom_2', 'Prec_ModLogNorm', 'Prec_AtenEspLivre']

# var_entrada = ['altura_tx', 'freq_tx', 'd_radial', 'class_autom_2']
# var_saida = ['Pr']
#
''' --- As variáveis do último estudo, a seguir, são as utilizadas para o prosseguimento
da avaliação --- '''
#
' -------- Estudo 1 -------- '
var_entrada = ['altura_tx', 'freq_tx',
               'dist_rota_desv_len_somb_pathloss_mseg'] #, 'class_autom_2']
var_saida = ['potRecNP_desv_len_somb_pathloss_mseg'] #, 'class_autom_2']

var_x = ['dist_rota_desv_len_somb_pathloss_mseg']
var_y = ['potRecNP_desv_len_somb_pathloss_mseg']

var_comparacao = ['Prec_ModLogNorm']
var_comparacao_dist = ['dist_rota']
#
#
#
' -------- Estudo 2 -------- '
var_entrada = ['altura_tx', 'freq_tx',
                'dist_rota'] #, 'class_autom_2']
var_saida = ['Prec_ModLogNorm'] #, 'class_autom_2']

var_x = ['dist_rota']
var_y = ['Prec_ModLogNorm']

var_comparacao = ['Prec_ModLogNorm']
var_comparacao_dist = ['dist_rota']
#
#
#
' -------- Estudo 3 -------- '
var_entrada = ['altura_tx', 'freq_tx',
                'dist_rota'] #, 'class_autom_2']
var_saida = ['Prec_AtenEspLivre'] #, 'class_autom_2']

var_x = ['dist_rota']
var_y = ['Prec_AtenEspLivre']

var_comparacao = ['Prec_AtenEspLivre']
var_comparacao_dist = ['dist_rota']
#
#
#
' -------- Estudo 4 -------- '
var_entrada = ['altura_tx', 'freq_tx',
               'dist_rota_desv_len_somb_pathloss_mseg'] #, 'class_autom_2']
var_saida = ['potRecNP_desv_len_somb_pathloss_mseg'] #, 'class_autom_2']

var_x = ['dist_rota_desv_len_somb_pathloss_mseg']
var_y = ['potRecNP_desv_len_somb_pathloss_mseg']

# var_comparacao = ['Pr']
# var_comparacao_dist = ['d_radial']
var_comparacao = ['Prec_ModLogNorm']
var_comparacao_dist = ['dist_rota']
#
'--------------------------------'
#
#
# ' -------- Estudo 5 -------- '
# #
# var_entrada = ['altura_tx', 'freq_tx',
#                 'dist_rota_desv_len_somb_pathloss_mseg',
#                 'classificacao'] #, 'class_autom_2']
# var_saida = ['potRecNP_desv_len_somb_pathloss_mseg'] #, 'class_autom_2']

# var_x = ['dist_rota_desv_len_somb_pathloss_mseg']
# var_y = ['potRecNP_desv_len_somb_pathloss_mseg']

# # var_comparacao = ['Pr']
# # var_comparacao_dist = ['d_radial']
# var_comparacao = ['Prec_ModLogNorm']
# var_comparacao_dist = ['dist_rota']
# #
# '--------------------------------'
#
print('\n---------------------------------------------------------------------')
print('Variáveis de entrada: ')
print(var_entrada)
print('\n---------------------------------------------------------------------')
print('Variável(is) de saída(s): ')
print(var_saida)
print('---------------------------------------------------------------------')
#
# Manutenção ou não das variáveis padrão
# --------------------------------------
entrada = input('Manter as variáveis de entrada e saída? [s] ')
if len(entrada) == 0 or entrada == 's':
    pass
else:
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
    # Escolha das variáveis de saída
    # ------------------------------
    var_saida=[]
    while True:
        lista = input('\n Entre número da posição da variável de saída: ');
        if len(lista) == 0:
            break
        var_saida.append(cab_colunas_rota_01[int(lista)])
    #
    print('\n---------------------------------------------------------------------')
    print('Variável(is) de saída(s): ')
    print(var_saida)
    print('---------------------------------------------------------------------')
#
#%% Desenho dos gráficos escolhidos
'------------------------------------------------------------------------------'
#
print("\014") # Clear console
# Variáveis padrão
# # ----------------
# var_x = ['d_radial']
# var_y = ['Pr']

print('\n---------------------------------------------------------------------')
print('Variável do eixo x: ' + var_x[0])
print('\n---------------------------------------------------------------------')
print('Variável do eixo y: ' + var_y[0])
print('---------------------------------------------------------------------')
#
# Manutenção ou não das variáveis padrão
# --------------------------------------
entrada = input('Manter as variáveis x e y? [s] ')
if len(entrada) == 0 or entrada == 's':
    pass
else:

    # Escolha da variável x - distância
    #----------------------------------
    escolha_col = ''
    conta_pos_col = 0
    for element in var_entrada:
        print(str(conta_pos_col) + ' - ' + element)
        conta_pos_col +=1
    
    var_x=[]
    while True:
        lista = input('\n Entre número da posição da variável x: ');
        if len(lista) == 0:
            break
        var_x.append(var_entrada[int(lista)])
    #
    # Escolha da variável y - amplitude
    #----------------------------------
    escolha_col = ''
    conta_pos_col = 0
    for element in var_saida:
        print(str(conta_pos_col) + ' - ' + element)
        conta_pos_col +=1
    
    var_y=[]
    while True:
        lista = input('\n Entre número da posição da variável y: ');
        if len(lista) == 0:
            break
        var_y.append(var_saida[int(lista)])
    
    # Escolha das variáveis para o gráfico
    # ------------------------------------
    print(var_y) # var_saida    # -->> var_atenuacao / var_potencia
    print(var_x) # var_entrada  # -->> var_distancia 

# %% Desenho dos gráficos escolhidos
# ----------------------------------
#
marcas_graf = ['.', 'o', 'v', 's', 'd', '^', 'x', 'p', 'h', '*']
for val_conta, conta_med in enumerate(medicoes_rota):

    # print(conta_med, conta_nome)
    
    # dist_pot_rec = locals()[conta_med + comp_nome_4]['dist_rota'].values
    dist_pot_rec = locals()[conta_med][var_x].values
    dist_pot_rec = dist_pot_rec[~np.isnan(dist_pot_rec)]

    # aten_pot_rec = locals()[conta_med + comp_nome_4]['Atenua_sinal_pot_rec'].values
    aten_pot_rec = locals()[conta_med][var_y].values
    aten_pot_rec = aten_pot_rec[~np.isnan(aten_pot_rec)]
    
    # ---------------------------------------------------
    if len(aten_pot_rec) == len(dist_pot_rec):
        pass
    elif len(aten_pot_rec) > len(dist_pot_rec):
        aten_pot_rec = aten_pot_rec[0:len(dist_pot_rec)]
    elif len(aten_pot_rec) < len(dist_pot_rec):
        dist_pot_rec = dist_pot_rec[0:len(aten_pot_rec)]
    # ---------------------------------------------------
    
    var_comp_y =  locals()[conta_med][var_comparacao].values
    var_comp_x =  locals()[conta_med][var_comparacao_dist].values 
    
    # Referencia log-distancia
    '-------------------------'
    # plt.plot(var_comp_x, var_comp_y, lw = 1, ls='--', c='k')
    
    # Sinal medido
    legenda_med = conta_med
    plt.plot(dist_pot_rec, aten_pot_rec,
             marker = marcas_graf[val_conta], markersize=6, markevery=500,
             label = legenda_med )

    # ENTER = input('\nPressione [ENTER] para continuar...\n')
    # plt.grid()
    # plt.show()

    # plt.text(dist_pot_rec[0], aten_pot_rec[0],
    #           conta_med, c = 'b', fontsize  = 10)

plt.title('Perda de percurso') #('Signal attenuation')
plt.xlabel('Distância - m')         # 'Distância - km'
plt.ylabel('Atenuação do sinal - dB')    # 'Nível do sinal - dB'

plt.legend(loc='upper center',ncol = 5, bbox_to_anchor = (0.5,-0.15),
            fontsize = 8 ) #bbox_to_anchor = (0,0),
plt.grid()
plt.show()
#
#%% Entrada de dados para a RNA
'------------------------------------------------------------------------------'
#
medicoes_estudo_total = pd.concat([medicoes_estudo_df[var_entrada],
                                  medicoes_estudo_df[var_saida]], axis = 1)
medicoes_estudo_total = medicoes_estudo_total.dropna(axis = 0)

medicoes_estudo_total = medicoes_estudo_total.sort_index().reset_index(drop=True)

# Dados de entrada
''' -------------- '''
matriz_val_ent = medicoes_estudo_total[var_entrada].values

# Eliminação de linha com nan
matriz_val_ent = matriz_val_ent[~np.isnan(matriz_val_ent).any(axis=1)]
#
# Dados de saída
''' ------------ '''
matriz_val_saida = medicoes_estudo_total[var_saida].values
#
# Eliminação de linha com nan
matriz_val_saida = matriz_val_saida[~np.isnan(matriz_val_saida).any(axis=1)]

# Matrizes de entradae saída
''' ------------------------------------------------ '''
X = matriz_val_ent
y = matriz_val_saida
#
# Verificaçãoe ajuste da dimensão das variáveis
''' ------------------------------------------- '''
# dimensao = shape(arquivoPlanilhaDados)
dimensao = np.shape(X)
numDim = np.shape(dimensao)
#
if numDim[0]<2:
    X = X.values.reshape(-1,1)
#
#%% Divisão dos dados em teste e treinamento
'------------------------------------------------------------------------------'
#
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.40,
                                                    random_state = 42)
#
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#
#%% Normalização e escalonamento dos dados
'------------------------------------------------------------------------------'
#
escala = StandardScaler()
X_train_escala = escala.fit_transform(X_train)
X_test_escala = escala.transform(X_test)

# escala_mm =  MinMaxScaler()
# X_train_escala = escala_mm.fit_transform(X_train)
# X_test_escala = escala_mm.transform(X_test)
#
#%% Criação da rede neural artificial - RNA
'------------------------------------------------------------------------------'
#
'''
referências:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlpregressor#sklearn.neural_network.MLPRegressor
https://www.springboard.com/blog/data-science/beginners-guide-neural-network-in-python-scikit-learn-0-18/
'''
#
# print("\033[H\033[J") # Clear console
print("\014")         # Clear console
#
proc_rna = ''
while not proc_rna:
    #
    # -----------------------------------      
    arranjo_rede_val = ''
    arranjo_rede = [] 
    while True:
        '---------------------------------------------'
        lista = input('Entre com o arranjo da rede:');
        if len(lista) == 0:
            break
        arranjo_rede_val = arranjo_rede_val + '_' + str(lista)
        arranjo_rede.append(int(lista))
    # -----------------------------------
    #      
    num_entradas = len(var_entrada)
    num_saidas = len(var_saida)#
    
    print("\014") # Clear console
    
    num_epocas = 1000
    # numEnt = 1
    cam_ocultas = np.array(arranjo_rede) # [5, 10]
    taxa_aprend = 0.00025 # 0.0000025 #  0.000025
    tolerancia = 0.00175 # 0.001 0.00000000001 #  0.000001 # 0.0005 0.001
    
    solucionador = 'adam' # 'adam'    #  'sgd'
    ativacao = 'relu'       # {'identity', logistic, 'tanh', 'relu'}, default=’relu’
    taxa_aprendizado = 'adaptive'

    '''
    reg = MLPRegressor(hidden_layer_sizes=[NUM_HIDDEN], 
                       max_iter=NUM_EPOCHS,
                       learning_rate_init=LR,
                       random_state=42)reg.fit(X_train, y_train)
    '''
    # Arquitetura da rede e hiperparâmetros
    rna = MLPRegressor(hidden_layer_sizes = cam_ocultas,     # Arquitetura: duas camadas de 10 e 5 neurônios
                        max_iter = num_epocas,               # hiperparâmetro -> quantidade de épocas
                        tol = tolerancia,                    # hiperparâmetro
                        learning_rate_init = taxa_aprend,    # hiperparâmetro
                        solver = solucionador,               # estratégia: Stocastic Gradient Descendent
                        activation = ativacao,               # função de ativação
                        learning_rate = taxa_aprendizado,    # taxa de aprendizagem: 'constant'
                        verbose = True)            
    #
    #% Parâmetros da rede neural
    '-----------------------------------------------------------------------'
    #
    print(rna)
    print(rna.get_params())
    for conta_par in rna.get_params():
        print(conta_par+ ': ', rna.get_params()[conta_par])
    #
    #% Treinamento RNA
    '-----------------------------------------------------------------------'
    #
    rna.fit(X_train_escala, y_train)
    #
    #% Avaliação da predição com as amostras de teste
    '-----------------------------------------------------------------------'
    # Acurácia
    # --------
    resultado = rna.score(X_test_escala, y_test)
    print('\n-----------------------------------\n')
    print('   Acuracia: %.2f%%' % (resultado*100.0))
    print('   Rede MPL', cam_ocultas)
    print('\n-----------------------------------\n')
    '''
    https://www.pluralsight.com/guides/validating-machine-learning-models-scikit-learn
    '''
    #
    #% Pesos / coeficientes da RNA
    '-----------------------------------------------------------------------'
    #
    pesosRNA = rna.coefs_
    biasRNA = rna.intercepts_
    print('Pesos', pesosRNA, '\nBias', biasRNA)
    #
    plt.plot(rna.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    #
    proc_rna = input('\nContinua processamento? s/[n]: ')
    if len(proc_rna) == 0 or proc_rna == 'n':
           proc_rna = 'n'
    else:
        break
    #
#%% Sumário da rede e visualização
'---------------------------------'
# @todo
# # ann_viz(rna, title="My first neural network")
# # Exemplo
# # ---------
# # import VisualizeNN as VisNN
# # network=VisNN.DrawNN([3,4,1])
# # network.draw()
# #
# import VisualizeNN as VisNN
# network=VisNN.DrawNN(arranjoDaRede)
# network.draw()
# #

#%% Gravação do modelo da rede neural
'------------------------------------------------------------------------------'
#
# Gravação para processamento atual
'----------------------------------'
#
ctrl_nome_arq = 0
while ctrl_nome_arq==0:
    #
    nome_grava_arq = input('\n\nNome padrão do arquivo [s]/n ?')
    if len(nome_grava_arq)==0 or nome_grava_arq.lower() == 's':
        #
        nome_arq = 'Rota_' + str(controla_escolha_rede_treina)
        ctrl_nome_arq = 1
        #
    elif nome_grava_arq.lower() == 'n':
        #
        nome_arq =  input('\n\nEntre nome do arquivo (Modelo_Treina_R1R2_):')
        ctrl_nome_arq = 1
        #
    else:
        nome_rota_proc = 'Rota_' + str(controla_escolha_rede_treina)
        nome_grava_arq = 0
#
# ---------------------------------------
num_entradas = len(var_entrada)

caminho_1 = 'Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/'
caminho_2 = 'MedicoesProcessados/ModeloRNA_ScikitLearn/'

caminho_grava_modelo_rna = caminho_1 + caminho_2

nome_arq_rna = nome_arq + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '.sav' # + '_' + data_hora_gravacao
#
# Gravação do arquivo
'--------------------' 
local_salvamento = caminho_grava_modelo_rna + nome_arq_rna
pickle.dump(rna, open(local_salvamento, 'wb'))

# Gravação para processamento posterior e registro final da rede
'---------------------------------------------------------------'
# data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")
data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm")

nome_arq_rna = nome_arq  + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '_' + data_hora_gravacao + '.sav'
#
# Gravação do arquivo
'--------------------'                
local_salvamento = caminho_grava_modelo_rna + nome_arq_rna
pickle.dump(rna, open(local_salvamento, 'wb'))
#
#%% Leitura do modelo escolhido da rede neural
'------------------------------------------------------------------------------'
#
# Rota_1_Rede_3_38_1.sav
# Rota_1_Rede_3_48_1.sav

# controla_escolha = '1'
# num_entradas = '3'
# arranjo_rede = [8, 5] # =--->> Ajustar de acordo com a rede que será processa a seguir
num_saidas = '1'

caminho_1 = 'Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/'
caminho_2 = 'MedicoesProcessados/ModeloRNA_ScikitLearn/'

caminho_leitura_modelo_rna = caminho_1 + caminho_2

# Para leitura do arquivo gravado anteriormente
'-------------------------------------------------'
nome_arq_rna = nome_arq + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '.sav'
#
# Para leitura diretamente do arquivo do diretório
'-------------------------------------------------'
nome_arq_rna = 'Rota_1_Rede_3_85_1.sav'
#
local_leitura = caminho_leitura_modelo_rna + nome_arq_rna
print(local_leitura)

rna = pickle.load(open(local_leitura, 'rb'))

result = rna.score(X_test_escala, y_test)
print(result)
#
#%% Comparação com o modelo de referência log-distância
'------------------------------------------------------------------------------'
#
print("\014") # Clear console
try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='inline') # 'inline') qt5
except:
    pass
#
# Escolha da rota de avaliação
'-----------------------------'
#
controla_escolha = 0
controla_escolha_avalia = 0
while controla_escolha_avalia==0:
    #
    nome_rota_proc = input("\n\nEntre com a rota para processamento: [1] / 2:\n")
    if len(nome_rota_proc)==0 or nome_rota_proc == '1':
        nome_rota_proc = '1'
        medicoes_rota = medicoes_rota_01
        # cab_colunas = cab_colunas_rota_01
        controla_escolha_avalia = 1
    elif nome_rota_proc == '2':
        medicoes_rota = medicoes_rota_02
        # cab_colunas = cab_colunas_rota_02
        controla_escolha_avalia = 2
    else:
        nome_rota_proc = ''
        controla_escolha_avalia = 0
#
# Variáveis para comparação
''' --------------------------------------------------- '''
var_comparacao
var_comparacao_dist
#
''' ---------------------------- '''
var_entrada
var_saida
#
# Variáveis de referência
' ----------------------- '
var_y
var_x
#
# Ajuste nas quantidade de amsotras para permitir comparação e cálculos
' ---------------------------------------------------------------------'
# var_saida_entrada_cab = var_saida + var_entrada

# var_saida_entrada_df = locals()[conta_med][var_saida_entrada_cab]
# var_saida_entrada_df = var_saida_entrada_df.dropna()

# X_predicao = var_saida_entrada_df[var_entrada].values
# y_real = var_saida_entrada_df[var_saida].values
#
#
var_ent_sai_comp = var_saida + var_entrada + var_comparacao + var_comparacao_dist
# Retirada de duplicadas
# print(dict.fromkeys(var_ent_sai_comp))
# print(list(dict.fromkeys(var_ent_sai_comp)))
var_ent_sai_comp = list(dict.fromkeys(var_ent_sai_comp))

var_ent_sai_comp_df = locals()[conta_med][var_ent_sai_comp]
var_ent_sai_comp_df = var_ent_sai_comp_df.dropna()

X_predicao = var_ent_sai_comp_df[var_entrada].values
y_real = var_ent_sai_comp_df[var_saida].values

log_distance = var_ent_sai_comp_df[var_comparacao].values
dist_log_distance = var_ent_sai_comp_df[var_comparacao_dist].values
' ---------------------------------------------------------------------'
#
print("\014") # Clear console
#
conta_rota_processada = 0
avalia_modelo_rna = ''

for conta_med in medicoes_rota:
# while not avalia_modelo_rna:
    #
    avalia_modelo_rna = input('\nAvaliar modelo da RNA [s]: ')
    # print("\014") # Clear console
    if len(avalia_modelo_rna) == 0:
        #
        '-----------------------------------------------------------------'
        # arq_medicao = input('Nome do arquivo para comparação [R1H1F1]:')
        # if len(arq_medicao) == 0:
        #     arq_medicao = 'R1H1F1'
        #     medicao = arq_medicao + '_df_proc'
        # else:
        #     medicao = arq_medicao + '_df_proc'
        # #
        '-----------------------------------------------------------------'
        medicao = conta_med #+ '_df_proc'
        print('Arquivo em procesaento -->> ', conta_med)
            
        # # Valores de entrada - Predição:
        # ''' ---------------------------- '''
        # # var_entrada
        # # var_saida
        
        # X_predicao = locals()[medicao][var_entrada].values
        # X_predicao = X_predicao[~np.isnan(X_predicao).any(axis=1)] # Eliminação de linha com nan
        
        # # Obtem dimensao da matriz p/ compatibilizar quant amostras entrada e saída
        # lin_pred, col_pred = X_predicao.shape
        
        # # Valores de saída - Real:
        # ''' ---------------------- '''
        # y_real = locals()[medicao][var_saida].values
        # y_real = y_real[~np.isnan(y_real).any(axis=1)] # Eliminação de linha com nan
        
        # # Ajusta quantidade de linhas para compatibiliar quant amostras entrada e saída
        # y_real = y_real[0:lin_pred,:]
        
        var_ent_sai_comp = var_saida + var_entrada + var_comparacao + var_comparacao_dist
        var_ent_sai_comp = list(dict.fromkeys(var_ent_sai_comp)) # Remove duplicdas
        var_ent_sai_comp_df = locals()[conta_med][var_ent_sai_comp]
        var_ent_sai_comp_df = var_ent_sai_comp_df.dropna()
        
        X_predicao = var_ent_sai_comp_df[var_entrada].values
        y_real = var_ent_sai_comp_df[var_saida].values
        

        # Previsão resultado aplicando a RNA
        '-----------------------------------'
        # Transformação de escala
        X_pred = escala.transform(X_predicao)
        #
        # Predição
        y_pred = rna.predict(X_pred) 
        #
        #% Avaliação da acurácia
        '-----------------------------------'
        #
        mae_rna = mean_absolute_error(y_pred, y_real)
        mse_rna = mean_squared_error(y_pred, y_real)
        rmse_rna = np.sqrt(mean_squared_error(y_pred, y_real))
        
        print('Mean Absolute Error - RNA: %1.2f' %mae_rna)
        print('Mean Squared Error - RNA: %1.2f' %mse_rna)
        print('Root Mean Squared Error - RNA: %1.2f' %rmse_rna)
        
        #
        #% Comparação com referência -> log-distância
        '-----------------------------------'
        #
        # Valores individuais:
        '-----------------------------------'
        n = locals()[medicao]['n'].values[0]
        p_c = locals()[medicao]['p_c'].values[0]
    
        n_aten_sinal = locals()[medicao]['n_aten_sinal'].values[0]
        p_c_aten_sinal = locals()[medicao]['p_c_aten_sinal'].values[0]
        
        n_aval_ref = n
        p_c_aval_ref = p_c
        
        # Valores processados e para comparação:
        '-----------------------------------'
        # log_distance = locals()[medicao][var_comparacao].values
        # log_distance = log_distance.reshape(-1,1)
        # log_distance = log_distance[~np.isnan(log_distance).any(axis=1)] # Eliminação de linha com nan
        # #        
        # dist_log_distance = locals()[medicao][var_comparacao_dist].values
        # dist_log_distance = dist_log_distance.reshape(-1,1)
        # dist_log_distance = dist_log_distance[~np.isnan(dist_log_distance).any(axis=1)] # Eliminação de linha com nan

        # dist_p_r = dist_log_distance
        # dist_p_r = locals()[medicao][var_x].values

        # Ajusta quantidade de linhas para compatibiliar
        # dist_p_r = dist_p_r[0:lin_pred]

        log_distance = var_ent_sai_comp_df[var_comparacao].values
        dist_log_distance = var_ent_sai_comp_df[var_comparacao_dist].values
        
        dist_p_r = dist_log_distance

        p_r = p_c_aval_ref + 10 * np.log10(dist_p_r/dist_p_r[0]) * n_aval_ref
        
        rmse_sinal_ld = np.sqrt(mean_squared_error(p_r, y_real))
        print('Root Mean Squared Error - LD: %1.2f' %rmse_sinal_ld)

        # '------------'
        # # ou ou ou ou 
        # '------------'
        
        # p_r = locals()[medicao]['Prec_ModLogNorm'].values
        # p_r = p_r[~np.isnan(p_r)]
        
        # '-------------------------------------------------'
        # comp_var_p_r = len(p_r)
        # comp_var_y_real = len(y_real)
        # if comp_var_p_r >= comp_var_y_real:
        #     p_r = p_r[0:comp_var_y_real]      
        # else:
        #     y_real_aten_ld = y_real[0:comp_var_p_r]
        # '-------------------------------------------------'
        # dist_p_r = dist_p_r[0:comp_var_p_r]   
        
        # rmse_sinal_ld = np.sqrt(mean_squared_error(p_r, y_real_aten_ld))

        # print('Root Mean Squared Error: %1.2f' %rmse_sinal_ld)
        
        #% Desenho - gráfico
        '-----------------------------------'
        #
        # Obtenção da distância real para utilização nos gráficos
        # --------------------------------------------------------
        X_dist_m = locals()[medicao][var_x].values
        X_dist_m = X_dist_m.reshape(-1,1)
        X_dist_m = X_dist_m[~np.isnan(X_dist_m).any(axis=1)] # Eliminação de linha com nan
        # --------------------------------------------------------
        #
        # Ajusta quantidade de linhas para compatibiliar
        # X_dist_m = X_dist_m[0:lin_pred,:]
        X_dist_m = var_ent_sai_comp_df[var_x].values
        # --------------------------------------------------------
        #
        # Títulos em português
        '---------------------'
        plt.title('Nível do sinal recebido - '+ medicao + \
                  '\nRNA - ' + str(arranjo_rede) + \
                  ' modelo: ' + str(controla_escolha_rede_treina) + \
                  ' / LD; n: %1.2f ' %n_aten_sinal) # 'Nível do sinal recebido'
        #
        # Títulos em inglês
        '-------------------'
        # plt.title('Received signal - '+ medicao + \
        #               '\nANN - ' + str(arranjo_rede) + \
        #               ' / Log-Dist; n: %1.2f ' %n_aten_sinal) # 'Nível do sinal recebido'
        '---------------------------------'
        plt.xlabel('Distância (m)')          # 'Distância - km'
        plt.ylabel('Sinal recebido (dBm)')    # 'Nível do sinal - dB'
        #
        # plt.scatter(X_dist_m, y_real, c = 'b',  s = 0.25)
        # plt.scatter(X_dist_m, y_pred, c = 'g',  s = 0.25)
        #
        # plt.plot(X_dist_m, y_real, c = 'b') #,  s = 0.25)
        plt.plot(X_dist_m, y_real,
                 c = 'b',
                 marker='*',
                 markersize=10,
                 markevery= 500)
        #
        # plt.scatter(X_dist_m, y_pred, c = 'g', s = 1, marker='^')
        plt.plot(X_dist_m, y_pred,
                 c = 'g', ls = '--', lw = 1,
                 marker='s',
                 markersize=7.5,
                 markevery= 500)
        #
        # Log-distância
        # -------------
        posEixoX = X_dist_m.min()
        posEixoY = X_dist_m.max()
        
        # plt.plot(dist_p_r, p_r, c = 'k', ls = '--')
        plt.plot(dist_p_r, p_r,
                 c = 'r', ls = '--', lw = 1,
                 marker='.',
                 markersize=10,
                 markevery= 500)
        # plt.scatter(dist_p_r, p_r, c = 'k', ls = '--', s = 0.25)
        # plt.scatter(dist_log_distance, log_distance, c = 'r',  s = 0.25)
        
        plt.legend([
                    'Medido',
                    'RNA x medido - RMSE  %1.2f' %rmse_rna,
                    'LD x medido - RMSE  %1.2f' %rmse_sinal_ld
                    ],
                    loc=3)

        plt.annotate('ANN - ' + str(arranjo_rede),
                      xy=(posEixoX,posEixoY), rotation=0, fontsize=8,
                      color = 'blue')
        #
        plt.grid()
        plt.show()
        #
        # Formação de tabelas para análise
        '-----------------------------------'
        if conta_rota_processada == 0:
            tabele_nomes = ['medicao',
                            'Arranjo da rede',
                            'n_aten_sinal',
                            'rmse_rna',
                            'rmse_sinal_ld'
                            ]
            
            tabela_dados = [medicao,
                            arranjo_rede,
                            n_aten_sinal,
                            rmse_rna,
                            rmse_sinal_ld
                            ]
            
            # Cria DataFrame vazio
            tabela_dados_df = pd.DataFrame(columns = tabele_nomes)
            # Adiciona linhas
            tabela_dados_df.loc[len(tabela_dados_df)] = tabela_dados
            
        else:
            tabela_dados = [medicao,
                            arranjo_rede,
                            n_aten_sinal,
                            rmse_rna,
                            rmse_sinal_ld
                            ]
            # Adiciona linhas
            tabela_dados_df.loc[len(tabela_dados_df)] = tabela_dados
        #
        
        
        conta_rota_processada +=1
        '-----------------------------------'
    else:
        avalia_modelo_rna = 1
#
#%% Gravação do arquivo com tabela de dados
'------------------------------------------------------------------------------'
#
caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/"

nome_rota_proc

# data_hora_gravacao = time.strftime("data_%d_%m-%H_%M_%S")
data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")

pasta_excel = 'mod_treina_rota_' + str(controla_escolha_rede_treina)
nome_arq_excel = 'ANN_rota_' + str(controla_escolha_rede_treina) + \
                     '_RNA' + arranjo_rede_val + '_' + data_hora_gravacao

nome_arq_xlsx = nome_arq_excel + '_SCiKitLearn' + '.xlsx'

# Gravação do arquivo
'--------------------'
tabela_dados_df.to_excel(caminho + nome_arq_xlsx,
                                sheet_name = pasta_excel,
                                index = False)
#
#%% Simulação de valores
'------------------------------------------------------------------------------'
#
print("\014") # Clear console
#
# quantidade de amostras de teste
quant_amostras = len(locals()[medicoes_rota[0]]['freq_tx'].values)

# Variáveis de teste
' ------------------'
var_ent_teste = ['alt_tx', 'freq_tx', 'dist_rota']
prefixo = '_teste'
var_entrada_teste = [nome_var + prefixo for nome_var in var_ent_teste]
#
# Variável 1 -> freq_tx
freq_tx_ini = 500e6
freq_tx_passo = 500e6
freq_tx_fim = 5e9 + freq_tx_passo
#
frequencia_tx =  [freq_tx for freq_tx in np.arange(freq_tx_ini,
                                                   freq_tx_fim,
                                                   freq_tx_passo)]
#
frequencia_tx = [705e6, 1790e6, 2.4e9, 3.5e9, 4e9 ]
frequencia_tx = [4e9]

#
# Variável 2 -> altura_tx
alt_ant_tx_ini = 1.7
alt_ant_tx_passo = 0.5
alt_ant_tx_fim = 4.2 + alt_ant_tx_passo
#
alturas_antena_tx = [alt_ant_tx for alt_ant_tx in np.arange(alt_ant_tx_ini,
                                                            alt_ant_tx_fim,
                                                            alt_ant_tx_passo)]
#
# alturas_antena_tx = [1.7, 4.2]
#
# Variável 3 -> dist_rota
dist_inicial= 40
dist_final = 70
dist_percurso = dist_final - dist_final
dist_rota_teste = np.linspace(dist_inicial, dist_final, quant_amostras)

# Variável n -> xxxxxx
class_autom_teste = locals()[medicoes_rota[0]]['class_autom_2'].values
#
# frequencia_tx = [705e6, 1790e6, 2.4e9, 3.5e9, 4e9 ]
#
# # Variáveis de referência
# ' ----------------------- '
# var_ref_x = 'dist_rota_desv_len_somb_pathloss_mseg'
# var_ref_y = 'potRecNP_desv_len_somb_pathloss_mseg'
#
#
for conta_alt_ant_tx in alturas_antena_tx:
    #
    for contf_freq_tx in frequencia_tx:
        #
        # Formação da tabela de teste
        ' --------------------------- '
        alt_tx_teste = np.repeat(conta_alt_ant_tx, quant_amostras)
        #
        freq_tx_teste = np.repeat(contf_freq_tx, quant_amostras)
        #
        '---------------------------------------------------------'
        # tabela_nomes_teste = var_entrada_teste
        # # tabela_dados_teste = np.array([alt_tx_teste, freq_tx_teste,
        # #                                dist_rota_teste]).T
        # #
        # tabela_dados_teste = np.array([alt_tx_teste, freq_tx_teste,
        #                                dist_rota_teste, class_autom_teste]).T

        # sinal_teste_df = pd.DataFrame(data = tabela_dados_teste,
        #                               columns = tabela_nomes_teste)
        '---------------------------------------------------------'
        #
        sinal_teste_df = pd.DataFrame()     # Cria dataframe vazio
        for conta_nome_col in var_entrada_teste:
            print(conta_nome_col)
            # input()
            sinal_teste_df =  pd.concat([sinal_teste_df,
                                         pd.DataFrame(locals()[conta_nome_col],
                                                      columns = [conta_nome_col])], axis = 1)
        #
        tabela_dados_teste = sinal_teste_df.to_numpy()
        #
        '---------------------------------------------------------'
        #        
        #% Predição com sinal de teste -> aplicação do modelos de RNA
        ''' --------------------------------------------------------- '''
        #
        X_predicao = tabela_dados_teste
        X_predicao = X_predicao[~np.isnan(X_predicao).any(axis=1)] # Eliminação de linha com nan
        
        # Previsão resultado aplicando a RNA
        '''---------------------------------'''
        X_pred = escala.transform(X_predicao)
        y_pred = rna.predict(X_pred) 
        #
        # Obtenção da distância real para utilizaçao nos gráficos
        ''' ----------------------------------------------------- '''
        X_dist_m = dist_rota_teste
        X_dist_m = X_dist_m.reshape(-1,1)
        X_dist_m = X_dist_m[~np.isnan(X_dist_m).any(axis=1)] # Eliminação de linha com nan
    
        #% Deseho dos gráficos escolhidos
        '--------------------------------'
        #
        # for conta_med, conta_nome in zip(nome_dados_rota, medicoes_rota):
        marcas_graf = ['.', 'o', 'v', 's', 'd', '^', 'x', 'p', 'h', '*']
        for conta_med in medicoes_rota_01:
    
            # print(conta_med, conta_nome)
            
            dist_pot_rec = locals()[conta_med][var_x].values
            dist_pot_rec = dist_pot_rec[~np.isnan(dist_pot_rec)]
        
            aten_pot_rec = locals()[conta_med][var_y].values
            aten_pot_rec = aten_pot_rec[~np.isnan(aten_pot_rec)]
        
            plt.plot(dist_pot_rec, aten_pot_rec,
                     marker = marcas_graf[val_conta], markersize=6, markevery=500,
                     label = conta_med)
            # plt.text(dist_pot_rec[-1], aten_pot_rec[-1],
            #          conta_med, c = 'b', fontsize  = 12)
        #
        valor_teste = 'H = %1.1f' %conta_alt_ant_tx + 'm - F = ' '%1.2e' %contf_freq_tx
        plt.plot(X_dist_m, y_pred,
                 lw = 2, ls = '--', c='k') # ,
                 # label = '%1.1f' %conta_alt_ant_tx + 'm \n' '%1.2e' %contf_freq_tx)
        #
        plt.title('Perda de percurso - rede: ' + str(cam_ocultas) + '\n'  + valor_teste)
        plt.xlabel('Distância - m')         # 'Distância - km'
        plt.ylabel('Atenuação do sinal - dB')    # 'Nível do sinal - dB'
        #
        # plt.legend(bbox_to_anchor = (1, 1), loc='upper left',
        #             fontsize = 8)
        plt.legend(loc='upper center',ncol = 5, bbox_to_anchor = (0.5,-0.15),
                    fontsize = 8 ) #bbox_to_anchor = (0,0),
        #
        plt.grid()
        plt.show()
        #
        # ------------------------------------------------------
        ENTER = input('\nPressione [ENTER] para continuar...\n')
        # ------------------------------------------------------
        #
#%% Geração de sinal de teste
'------------------------------------------------------------------------------'
#
print("\014") # Clear console
#
class_autom_teste = locals()[medicoes_rota[0]]['class_autom'].values

quant_amostras = len(class_autom_teste)

# Variáveis de teste
' ------------------'
var_ent_teste = ['alt_tx', 'freq_tx', 'dist_rota']
prefixo = '_teste'
var_entrada_teste = [nome_var + prefixo for nome_var in var_ent_teste]

dist_inicial= 40
dist_final = 70
dist_percurso = dist_final - dist_final
dist_rota_teste = np.linspace(dist_inicial, dist_final, quant_amostras)

# Valores de teste
# ----------------
F1 = 705e6
F2 = 1790e6
F3 = 2400e6
F4 = 3500e6
F5 = 4000e6

FA = F1/2
FB = (F1 + F2)/2
FC = (F2 + F3)/2
FD = (F3 + F4)/2
# ----------------
frequencia_teste = F5 # F5 # (705e6 + 1790e6)/2
freq_tx_teste = np.repeat(frequencia_teste, quant_amostras)
# ----------------
#
proc_sinal_teste = ''
while not proc_sinal_teste:
    # print("\014") # Clear console
    proc_sinal_teste = input('\nContinua processamento? [s] / n :')
    if len(proc_sinal_teste) == 0 or proc_sinal_teste == 's':
           proc_sinal_teste = ''
    else:
        break
    #
    altura_tx_teste = [] 
    while True:
        altura_tx_teste = input('Entre com a altura antena:');
        if len(altura_tx_teste) == 0:
            altura_tx_teste = []
        else:
            altura_tx_teste = float(altura_tx_teste)
            break
    #
    freq_tx_teste = [] 
    freq_tx_teste_unica = 0
    while True:
        freq_tx_teste = input('Entre com a frequencia de teste:');
        if len(freq_tx_teste) == 0:
            freq_tx_teste = []
        else:
            freq_tx_teste = float(freq_tx_teste)
            freq_tx_teste_unica = int(freq_tx_teste)
            break
    #
    # Formação da tabela de teste
    ' --------------------------- '
    alt_tx_teste = np.repeat(altura_tx_teste, quant_amostras)
    #
    freq_tx_teste = np.repeat(freq_tx_teste_unica, quant_amostras)
    
    # freq_tx_teste = freq_tx_teste.reshape(-1,1)
    # alt_tx_teste = alt_tx_teste.reshape(-1,1)
    # dist_rota_teste = dist_rota_teste.reshape(-1,1)
    # class_autom_teste = class_autom_teste.reshape(-1,1)
    
    # tabela_nomes_teste = var_entrada_teste

    # # tabela_dados_teste = np.array([freq_tx_teste,
    # #                 alt_tx_teste,
    # #                 dist_rota_teste
    # #                 ]).T
        
    # tabela_dados_teste = np.concatenate((freq_tx_teste,
    #                 alt_tx_teste,
    #                 dist_rota_teste
    #                 ), axis = 1
    #                 )
    
    # sinal_teste_df = pd.DataFrame(data = tabela_dados_teste,
    #                               columns = tabela_nomes_teste)
    #
    sinal_teste_df = pd.DataFrame()     # Cria dataframe vazio
    for conta_nome_col in var_entrada_teste:
        print(conta_nome_col)
        # input()
        sinal_teste_df =  pd.concat([sinal_teste_df,
                                     pd.DataFrame(locals()[conta_nome_col],
                                                  columns = [conta_nome_col])], axis = 1)
    #
    tabela_dados_teste = sinal_teste_df.to_numpy()

    #% Predição com sinal de teste -> aplicação do modelos de RNA
    '------------------------------------------------------------'
    #
    X_predicao = tabela_dados_teste
    X_predicao = X_predicao[~np.isnan(X_predicao).any(axis=1)] # Eliminação de linha com nan
    
    # Previsão resultado aplicando a RNA
    '-----------------------------------'
    X_pred = escala.transform(X_predicao)
    y_pred = rna.predict(X_pred) 
    #
    # Obtenção da distância real para utilizaçao nos gráficos
    '--------------------------------------------------------'
    X_dist_m = dist_rota_teste
    X_dist_m = X_dist_m.reshape(-1,1)
    X_dist_m = X_dist_m[~np.isnan(X_dist_m).any(axis=1)] # Eliminação de linha com nan

    #% Deseho dos gráficos escolhidos
    '--------------------------------------------------------'
    #
    # for conta_med, conta_nome in zip(nome_dados_rota, medicoes_rota):
    for conta_med in medicoes_rota_01:

        # print(conta_med, conta_nome)
        
        dist_pot_rec = locals()[conta_med][var_x].values
        dist_pot_rec = dist_pot_rec[~np.isnan(dist_pot_rec)]
    
        aten_pot_rec = locals()[conta_med][var_y].values
        aten_pot_rec = aten_pot_rec[~np.isnan(aten_pot_rec)]
    
        plt.plot(dist_pot_rec, aten_pot_rec, label = conta_med)
        # plt.text(dist_pot_rec[-1], aten_pot_rec[-1],
        #          conta_med, c = 'b', fontsize  = 12)
    #
    plt.plot(X_dist_m, y_pred, '--k', label = str(altura_tx_teste) + 'm \n'
                                            '%1.2e' %freq_tx_teste_unica)
    #
    plt.title('Signal attenuation')
    plt.xlabel('Distance - m')         # 'Distância - km'
    plt.ylabel('Signal attenuation - dB')    # 'Nível do sinal - dB'
    #
    plt.legend(bbox_to_anchor = (1, 1), loc='upper left',
                fontsize = 8)
    plt.grid()
    plt.show()
#
#%% Gravaçao do modelo de predição
'------------------------------------------------------------------------------'
# Nome do arquivo para gravação (=> escolher)
#
nomeArquivo = 'rna_'
nomeArquivoGrava = nomeArquivo + "LogNormal"
#

caminhoModeloRNA = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/RedesNeurais/"
nomeDoArqModeloRNA = 'modelo_' + nomeArquivoGrava + '.sav' 
#
#
arquivoRNA = caminhoModeloRNA + nomeDoArqModeloRNA
pickle.dump(rna, open(arquivoRNA, 'wb'))
#
#
nomeDoArqModeloRNA = 'modelo_' + nomeArquivoGrava + '.pkl' 
arquivoRNA = caminhoModeloRNA + nomeDoArqModeloRNA
joblib.dump(rna, arquivoRNA)
#

# %% Testes e estudos
'------------------------------------------------------------------------------'
#
rota_med_1 = 'R2H1F5'
rota_med_2 = 'R1H2F1'

valor_x1 = locals()[rota_med_1]['Prec_AtenEspLivre'].values
valor_y1 = locals()[rota_med_1]['Pr'].values
valor_z1 = locals()[rota_med_1]['Prec_ModLogNorm'].values

# valor_x2 = locals()[rota_med_2]['Prec_AtenEspLivre'].values
# valor_y2 = locals()[rota_med_2]['Pr'].values
# valor_z2 = locals()[rota_med_2]['Prec_ModLogNorm'].values


# plt.plot(valor_x1, valor_x2)
# plt.plot(valor_y1, valor_y2)
# plt.plot(valor_z1, valor_z2)

plt.show()

proc_sinal_teste = ''
for conta_med in medicoes_rota_02:

    valor_y1 = locals()[rota_med_1]['potRecNP_desv_len_pathloss'].values
    valor_y2 = locals()[conta_med]['potRecNP_desv_len_pathloss'].values

    plt.plot(valor_y1, valor_y2)
    plt.show()
    proc_sinal_teste = input('\nContinua processamento? [s] / n :')
    if len(proc_sinal_teste) == 0 or proc_sinal_teste == 's':
           proc_sinal_teste = ''
    else:
        break


'------'
# @todo
'------'
# %% Avaliação do modelo - I
'------------------------------------------------------------------------------'

X_train, X_test, y_train, y_test 

escala = StandardScaler()
X_train_escala = escala.fit_transform(X_train)
X_test_escala = escala.transform(X_test)

res_treina = rna.score(X_train_escala, y_train)
print("Acurácia treinamento: %.2f%%" % (res_treina*100.0))

res_teste = rna.score(X_test_escala, y_test)
print("Acurácia teste: %.2f%%" % (res_teste*100.0))

'---------------------------------------------'


# res_treina = rna.score(X_train_escala, y_train)
# print("Accuracy: %.2f%%" % (res_treina*100.0))

# res_teste = rna.score(X_test_escala, y_test)
# print("Accuracy: %.2f%%" % (res_teste*100.0))

# %% Avaliação do modelo - II
'----------------------------'
#
'''
Ref.: https://visualstudiomagazine.com/Articles/2023/05/01/regression-scikit.aspx?Page=1
'''
def accuracy(model, data_x, data_y, pct_close=0.10):
  # accuracy predicted within pct_close of actual income
  # item-by-item allows inspection but is slow
  n_correct = 0; n_wrong = 0
  predicteds = model.predict(data_x)  # all predicteds
  for i in range(len(predicteds)):
    actual = data_y[i]
    pred = predicteds[i]

    if np.abs(pred - actual) < np.abs(pct_close * actual):
      n_correct += 1
    else:
      n_wrong += 1
  acc = (n_correct * 1.0) / (n_correct + n_wrong)
  return acc

# 4. evaluate model
'------------------'
print("Compute model accuracy (within 0.10 of actual) ")

acc_train = accuracy(rna, X_train_escala, y_train, 0.10)
print("Accuracy on train = %0.4f " % acc_train)

acc_test = accuracy(rna, X_train_escala, y_train, 0.10)
print("Accuracy on test = %0.4f " % acc_test)


# %% Avaliação feita utilizando contjunto de treinamento conjunto de teste
'-------------------------------------------------------------------------'
# 
# https://www.pluralsight.com/guides/validating-machine-learning-models-scikit-learn
#
y
y_1d = y.reshape(-1,)

escala = StandardScaler()
X_escala = escala.fit_transform(X)

res_treina_teste = rna.score(X_escala, y)
print("Accuracy: %.2f%%" % (res_treina_teste*100.0))

mostra_caclulo = True

# %% K-fold Cross-Validation
'---------------------------'
#
kfold = model_selection.KFold(n_splits=5, random_state=None)
# model_kfold = MLPRegressor()
# kfold = rna.KFold(n_splits=10, random_state=None)
# y_1d = y.reshape(-1,)
resultado_kfold = cross_val_score(rna, X_escala, y_1d,
                                  cv=kfold,
                                  verbose=mostra_caclulo)

print("Accuracy: %.2f%%" % (resultado_kfold.mean()*100.0))
resultado_kfold.std()
resultado_kfold

# %% Validação Cruzada Estratificada K-fold
'------------------------------------------'
'Somente para binario e multicalsse, não para contínuo'
# #
# skfold = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)
# # model_skfold = LogisticRegression()
# # model_skfold = MLPRegressor()
# results_skfold = cross_val_score(rna, X_escala, y_1d,
#                                  cv=skfold,
#                                  verbose=mostra_caclulo)

# print("Accuracy: %.2f%%" % (results_skfold.mean()*100.0))

# %% Deixar um de fora da validação cruzada (LOOCV)
'--------------------------------------------------'
#
loocv = model_selection.LeaveOneOut()
# model_loocv = LogisticRegression()
results_loocv = model_selection.cross_val_score(rna, X_escala, y_1d,
                                                cv=loocv,
                                                verbose=mostra_caclulo)

print("Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
      
# %% Validação cruzada - CROSS VALIDATION
'----------------------------------------'
#
pontuacaoRNA = cross_val_score(rna, X_escala, y,
                               cv=5,
                               scoring="neg_mean_squared_error",
                               verbose=mostra_caclulo)
pontuacaoRNA_RMSE = np.sqrt(-pontuacaoRNA)

print('\n------------------------------------------------- ')
print(" Pontuação RMSE: ",  pontuacaoRNA_RMSE)
print(' Média:', pontuacaoRNA_RMSE.mean())
print(' Desvio padrão:',pontuacaoRNA_RMSE.std())
print('------------------------------------------------- ')

# %%
'-------------------------------'
#
cv_dict = cross_validate(rna, X_escala, y,
                         return_train_score=True)
#
#
#%% Busca pelos melhores hiperparâmetros  RNA - Grid search
'------------------------------------------------------------------------------'
#
'''
REF.: https://michael-fuchs-python.netlify.app/2021/02/10/nn-multi-layer-perceptron-regressor-mlpregressor/
'''
# ===>>> Os avisos foram suprimidos -> ver importação Warning '''

'''
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
sorted(clf.cv_results_.keys())
'''
arranjoDaRede = [8]
#
RNA_parametros = {
    #
    # 'hidden_layer_sizes': [(9), (10), (11,11), (12), (13)], #[(11,11), (22,22), (33,33)],
    'hidden_layer_sizes': [(2, 6), (3, 7), (4, 8), (5, 8), (6, 9)], #[(11,11), (22,22), (33,33)],
        # np.array([8]),
        #                    np.array([16]),
        #                    np.array([32]),
        #                    np.array([8, 16]),
        #                    np.array([16, 16]),
        #                    np.array([16, 16, 16])], # 10, 20, 30]),
    'activation': ['relu'], # default='relu' 'identity', 'logistic', 'tanh', 
    'max_iter': [1000],
    'solver': ["adam"], # 'sgd', 'lbfgs', 'adam'],                    # default='adam'
    'tol': [0.0005], #, 0.001, 0.0015),
    'learning_rate_init': [0.001] # , 0.0001, 0.00015)
    }
# 'activation': ['identity', 'logistic', 'tanh', 'relu'], # default='relu' 
#
#%% GridSearch with cross validation
'------------------------------------------------------------------------------'

gridSearch = GridSearchCV(MLPRegressor(),RNA_parametros, cv=3,
                          scoring="neg_mean_squared_error",
                          verbose=3)
#
# gridSearch = RandomizedSearchCV(MLPRegressor(),RNA_parametros, cv=5,
#                           scoring="neg_mean_squared_error")
#
#%% Treinamaneto utilizando o Grid search
'------------------------------------------------------------------------------'
#
RNA_model = gridSearch.fit(X_escala, y_1d)
#
# RNA_model = gridSearch.fit( X_escala, y)
#
#%% Verificação dos melhores resultados e estimadores
'------------------------------------------------------------------------------'

print(RNA_model.best_params_,'\n')
print(RNA_model.best_estimator_,'\n')
#
#%% Verificação das pontuações de avaliação
'------------------------------------------------------------------------------'

resultadoCrossValidation = gridSearch.cv_results_
for mean_score, params in zip(resultadoCrossValidation["mean_test_score"],
                              resultadoCrossValidation["params"]):
    print(np.sqrt(-mean_score), params)
#
#%% Avaliação da importância dos parâmetros
'------------------------------------------------------------------------------'
#
# importanciaParametros = RNA_model.best_estimator_.feature_importances_
# AttributeError: 'MLPRegressor' object has no attribute 'feature_importances_'
#
'''
----------------
Final
==============================================================================
'''
