'''
# -*- coding: utf-8 -*-
-------------------------------------------------------------------------------
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
import os
import time

from sklearn.metrics import r2_score, \
                            mean_absolute_error, \
                            mean_squared_error, \
                            mean_squared_log_error, \
                            explained_variance_score
#
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, \
                                    GridSearchCV, \
                                    RandomizedSearchCV, \
                                    cross_val_score
#
from ann_visualizer.visualize import ann_viz
from colorama import Fore, Back, Style
#
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, \
                                        ReduceLROnPlateau, \
                                        ModelCheckpoint, \
                                        TensorBoard
from tensorflow.keras.utils import plot_model

from tensorflow.keras.backend import clear_session

# from tensorflow import keras

#%% Definição de funções
' ---------------------- '
#
#% Gravação do arquivo tabela de dados
' ------------------------------------ '
#
def grava_tabela_dados(tabela_nomes, tabela_dados_df, nome_rota_proc, sufixo_tabela):
    caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/"
    
    pasta_excel = 'rota_' + nome_rota_proc
    nome_arq_excel = 'tabCompModelRef_LogDistancia_rota_' + nome_rota_proc + sufixo_tabela
    nome_arq_xlsx = nome_arq_excel + '.xlsx'
    
    # Gravação do arquivo
    ''' ----------------- '''
    tabela_dados_df.to_excel(caminho + nome_arq_xlsx,
                                    sheet_name = pasta_excel,
                                    index = False)
#
#
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
medicoes_rota_total = medicoes_rota_01 + medicoes_rota_02
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
# %% Obtem lista com o cabeçalho das colunas
'------------------------------------------------------------------------------'
#
''' Cabeçalhos colunas:
----------------------- '''
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
#%%
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
# medicoes_estudo_df =  pd.concat([R1H1F1, R1H1F2, R1H1F3,
#                           R1H1F4, R1H1F5,
#                           R1H2F1, R1H2F2, R1H2F3,
#                           R1H2F4, R1H2F5]).sort_index().reset_index(drop=True)
#
# medicoes_estudo_df =  pd.concat([R1H1F1, R1H1F2, R1H1F3,
#                           R1H1F4, R1H1F5,
#                           R1H2F1, R1H2F2, R1H2F3,
#                           R1H2F4, R1H2F5])
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
#%% Escolha das variáveis de entrada e saída
'------------------------------------------------------------------------------'
#
print("\014") # Clear console
#
# Variáveis padrão
# ----------------
# var_entrada = ['altura_tx', 'freq_tx', 'd_radial', 'Prec_ModLogNorm']
# var_entrada = ['altura_tx', 'freq_tx', 'd_radial']

# var_entrada = ['altura_tx', 'freq_tx', 'd_radial', 'class_autom']
# var_entrada = ['altura_tx', 'freq_tx', 'd_radial', 'Prec_AtenEspLivre']
# var_saida = ['Pr']
#
#
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
# ----------------
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
# -------------------------------
for conta_med in medicoes_rota:

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
    plt.plot(dist_pot_rec, aten_pot_rec, label = conta_med)

    # ENTER = input('\nPressione [ENTER] para continuar...\n')
    # plt.grid()
    # plt.show()

    # plt.text(dist_pot_rec[-1], aten_pot_rec[-1],
    #          conta_med, c = 'b', fontsize  = 12)
    
plt.title('Signal attenuation')
plt.xlabel('Distance - m')         # 'Distância - km'
plt.ylabel('Signal attenuation - dB')    # 'Nível do sinal - dB'

plt.legend(loc='upper center',ncol = 5, bbox_to_anchor = (0.5,-0.15),
            fontsize = 8 ) #bbox_to_anchor = (0,0),
plt.grid()
plt.show()
#
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
escala_keras = StandardScaler() # MinMaxescala()
X_train_escala = escala_keras.fit_transform(X_train)
X_test_escala = escala_keras.transform(X_test)
#
# escala_mm =  MinMaxScaler()
# X_train_escala = escala_mm.fit_transform(X_train)
# X_test_escala = escala_mm.transform(X_test)
#
#%% Criação da rede neural artificial - RNA
'------------------------------------------------------------------------------'
#
print("\014")         # Clear console

clear_session()

# del modelo_rna

arranjo_rede_val = ''
arranjo_rede = [] 
while True:
    '---------------------------------------------'
    lista = input('Entre com o arranjo da rede: ');
    if len(lista) == 0:
        break
    arranjo_rede_val = arranjo_rede_val + '_' + str(lista)
    arranjo_rede.append(int(lista))
# -----------------------------------      
#
num_entradas = len(var_entrada)
num_saidas = len(var_saida)

entradas_zeradas = 0.2
ativacao = 'relu'
ativacao_saida = 'linear'
 
# Formação do modelo
'-------------------'
modelo_rna = Sequential()

# Camada de entrada e camada(s) oculta(s)
'----------------------------------------'
conta_valor_arranjo = 0
for valor_aranjo in arranjo_rede:
    print(valor_aranjo)
    #
    if conta_valor_arranjo == 0:
        #
        # Camada de entrada e 1a. camada oculta
        '-------------------------------------'
        modelo_rna.add(Dense(units = valor_aranjo,
                             input_shape = (num_entradas,),
                             kernel_initializer = 'he_uniform', # normal he_uniform
                             activation = ativacao))
        #
        conta_valor_arranjo += 1
        #
    else:
        # 2a. camada oculta em demais
        '-------------------------------------'
        modelo_rna.add(Dense(valor_aranjo,
                             activation = ativacao))
        # modelo_rna.add(Dropout(entradas_zeradas))
#
# Camada de saída
'----------------'
modelo_rna.add(Dense(units = num_saidas,
                      activation = ativacao_saida))
#
#%%
''' ---------------------------------------------------- '''
ENTER = input('\nPressione [ENTER] para continuar...\n')
''' ---------------------------------------------------- '''
#
# %% Compilação do modelo 
'------------------------------------------------------------------------------'
#
modelo_rna.compile(optimizer='adam', # SGD
              loss = 'mse',
              metrics = ['mean_absolute_error', 'accuracy'])
#
#%% Sumário da rede e visualização
'------------------------------------------------------------------------------'
#
modelo_rna.summary()
#
# %% Desenho arquitetura rede
'----------------------------'

caminho_1 = r'Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao'
caminho_2 = '\Artigo_03_Vegetacao\Programas\MedicoesProcessados'
caminho_3 = '\DesenhoRede'
caminho_total = caminho_1 + caminho_2 + caminho_3 

nome_arquivo = '\RedeNeural.png'
caminho_arquivo = caminho_1 + caminho_2 + caminho_3 + nome_arquivo
#
plot_model(modelo_rna,
           to_file= caminho_arquivo,
           show_shapes=True,
           show_layer_names=True)
#
#%% Sumário da rede e visualização - geração arquivo pdf
'------------------------------------------------------------------------------'
# BUG

'''
Ref.:
    https://www.youtube.com/watch?v=_OfhVK6w3xc&t=682s
'''
#
# caminho_1 = ''
# caminho_2 = r'E:\Doutorado'
#
# caminho_1 = r'Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao'
# caminho_2 = '\Artigo_03_Vegetacao\Programas\MedicoesProcessados'
# caminho_3 = '\DesenhoRede'
nome_arquivo = '\RedeNeural.gv'
caminho_arquivo = caminho_1 + caminho_2 + caminho_3 + nome_arquivo

try:
        # from ann_visualizer.visualize import ann_viz;
    # ann_viz(modelo_rna, view=True) # filename = caminho_arquivo, title="ANN Diagram"
    ann_viz(modelo_rna,
            filename = 'RedeNeural.gv', 
            title='')
    #
    # ann_viz(modelo_rna, view = True, title='')
except:
    pass

# %%
'-------------------------------------------------------'
# BUG

print('\014') # Clear console
print()
print('-------------------------------------------------------')
print('Feche o arquivo:' + nome_arquivo + '.pdf' + 'aberto no pdf Reader')
print('-------------------------------------------------------')
print()

ENTER = input('\nPressione [ENTER] para continuar...\n')
'-------------------------------------------------------'
#
os.remove(caminho_arquivo + '.pdf')

# %%
'-------------------------------------------------------'
# BUG

import graphviz
graph_file = graphviz.Source.from_file(caminho_arquivo)
graph_file.view()

#%% Treinamento do modelo
'------------------------------------------------------------------------------'
#
# Parâmetros callback
'-------------------------------------------------------------'
es = EarlyStopping(monitor = 'loss', min_delta = 1e-6,
                    patience = 3, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2,
                        patience = 3, verbose = 1)
# mcp = ModelCheckpoint(filepath = arqPesos, monitor = 'loss', 
#                       save_best_only = True, verbose = 1)
'-------------------------------------------------------------'
#
# Tensorboard - configuração
'---------------------------' 
caminho_1 = r'Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao'
caminho_2 = '\Artigo_03_Vegetacao\Programas\MedicoesProcessados'
caminho_3 = '\DirLogTensorBoard_Unico'
caminho_log = caminho_1 + caminho_2 + caminho_3

tensorboard_cb = TensorBoard(log_dir = caminho_log) #, histogram_freq = 1)

# Treinamneto
'-------------------------------------------------------------'
history = modelo_rna.fit(x=X_train_escala, y=y_train,
              validation_data=(X_test_escala, y_test),
              batch_size = 100,
              epochs = 50,
              callbacks = [tensorboard_cb, es, rlr])
              # callbacks = [EarlyStopping(monitor = 'val_loss', patience = 10)])
              # callbacks = [es, rlr, mcp])

# history = model.fit(
#    X_train, y_train,    
#    batch_size=128, 
#    epochs = 500, 
#    verbose = 1, 
#    validation_split = 0.2, 
#    callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]

# # ------------------------------------------------------
# ENTER = input('\nPressione [ENTER] para continuar...\n')
# # ------------------------------------------------------
#
'https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/'
#
perda, acuracia, _= modelo_rna.evaluate(X_test_escala, y_test, verbose = 0)
print('\n\n Perda do modelo [MSE]: %.2f, Métrica [mean_absolute_error]: %.2f'  % ((perda, acuracia)))
#
#%% Gráficos das perdas
'------------------------------------------------------------------------------'
#
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# #
# %% Descrisão domodelo obtido
'-----------------------------'
#
modelo_rna.get_weights()

print('Pesos e vieses das camadas do melhor modelo \n')
for layer in modelo_rna.layers:
    print(layer.name)
    print("-------------------------------------------")
    print("Pesos - Forma: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
    print("--------------------------------------------")
    print("Viés - Forma: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')
#
# %% Inicialização do Tensorboard
'--------------------------------'
#
import os
from threading import Thread
#
# funções
'-----------------------'
def tarefa_sistema(caminho_log):
    porta = 6006
    print ('\nIniciando tensor board')
    # comando = r'tensorboard --logdir Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao\Artigo_03_Vegetacao\Programas\MedicoesProcessados\KerasTuner --port ' + str(porta)
    # comando = r'tensorboard --logdir Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao\Artigo_03_Vegetacao\Programas\MedicoesProcessados\DirLogTensorBoard_Unico --port ' + str(porta)
    comando = r'tensorboard --logdir ' + caminho_log + ' --port ' + str(porta)
    comando = comando.replace('\\', '/')
    # print(comando)
    os.system(comando)
    print ('\n Tensor board finalizado')
#
os.system('npx kill-port 6006')

# Carregar tensorBoard
'---------------------'
# %load_ext tensorboard
# %reload_ext tensorboard
os.system('reload_ext tensorboard')


'-----------------------------------------------------'
# Cria Thread (tarefa)
tarefa_1 = Thread(target = tarefa_sistema, args=(caminho_log,))

# Inicia Thread
tarefa_1.start()

'-----------------------------------------------------'
import webbrowser
porta = 6006
url_tb = 'http://localhost:' + str(porta)

# Executa TansorBoard no Navegador Padrão
'----------------------------------------'
webbrowser.open(url_tb)

# %% Finaliza Tread ao finalizar o estudo
'----------------------------------------'
#
tarefa_1.join()

# %% Gravação do modelo da rede neural
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

# ---------------------------------------
num_entradas = len(var_entrada)

caminho_1 = 'Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/'
caminho_2 = 'MedicoesProcessados/ModelosRNA/'

caminho_grava_modelo_rna = caminho_1 + caminho_2

nome_arq_rna = nome_arq + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '.h5'
#
# Gravação do arquivo
'--------------------'                
modelo_rna.save(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))
#
# Gravação para processamento posterior e registro final da rede
'---------------------------------------------------------------'
data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")

nome_arq_rna = nome_arq  + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '_' + data_hora_gravacao + '.h5'
#
# Gravação do arquivo
'--------------------'                
modelo_rna.save(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))
#
# %% Leitura do modelo escolhido da rede neural
'------------------------------------------------------------------------------'
#
del modelo_rna

modelo_rna_carregado = load_model(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))

modelo_rna = modelo_rna_carregado

modelo_rna.get_weights()

#
#%% Comparação com o modelo de referência log-distância com rotas
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
# ---------------------------------------
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
# Escolha das variáveis para comparação
''' --------------------------------------------------- '''
var_comparacao
var_comparacao_dist
#
''' ---------------------------- '''
var_entrada
var_saida
#
''' ---------------------------- '''
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
        X_pred = escala_keras.transform(X_predicao)
        #
        # Predição melhor_modelo modelo_rna
        y_pred = modelo_rna.predict(X_pred) 
        #
        #% Avaliação da acurácia
        '-----------------------------------'
        #
        mae_rna = mean_absolute_error(y_pred, y_real)
        mse_rna = mean_squared_error(y_pred, y_real)
        rmse_rna = np.sqrt(mean_squared_error(y_pred, y_real))
        
        print('Mean Absolute Error: %1.2f' %mae_rna)
        print('Mean Squared Error: %1.2f' %mse_rna)
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
        plt.title('Received signal - '+ medicao + \
                  '\nANN - ' + str(arranjo_rede) + \
                  ' / Log-Dist; n: %1.2f ' %n_aten_sinal)        # 'Nível do sinal recebido'
        plt.xlabel('Distance - m')          # 'Distância - km'
        plt.ylabel('signal level - dBm')    # 'Nível do sinal - dB'
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
        # plt.scatter(X_dist_m, y_pred, c = 'g') #,  s = 0.25)
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
                    'Measured',
                    'ANN x measured - RMSE  %1.2f' %rmse_rna,
                    'LD x measured - RMSE  %1.2f' %rmse_sinal_ld
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
#
#%% Gravação da tabela de dados
'------------------------------------------------------------------------------'
#
caminho = r"Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/MedicoesProcessados/"

nome_rota_proc

data_hora_gravacao = time.strftime("data_%d_%m-%H_%M_%S")

pasta_excel = 'rota_' + nome_rota_proc
nome_arq_excel = 'ANN_rota_' + str(nome_rota_proc) + \
                     '_RNA' + arranjo_rede_val + data_hora_gravacao

nome_arq_xlsx = nome_arq_excel + 'Keras' + '.xlsx'

# Gravação do arquivo
'--------------------'
tabela_dados_df.to_excel(caminho + nome_arq_xlsx,
                                sheet_name = pasta_excel,
                                index = False)

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
frequencia_tx = [4e9 ]

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
# Variáveis de referência
' ----------------------- '
var_ref_x = 'dist_rota_desv_len_somb_pathloss_mseg'
var_ref_y = 'potRecNP_desv_len_somb_pathloss_mseg'
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
        '------------------------------------------------------------'
        #
        X_predicao = tabela_dados_teste
        X_predicao = X_predicao[~np.isnan(X_predicao).any(axis=1)] # Eliminação de linha com nan
        
        # Previsão resultado aplicando a RNA
        '-----------------------------------'
        X_pred = escala_keras.transform(X_predicao)
        y_pred = modelo_rna.predict(X_pred) 
        #
        # Obtenção da distância real para utilizaçao nos gráficos
        '--------------------------------'
        X_dist_m = dist_rota_teste
        X_dist_m = X_dist_m.reshape(-1,1)
        X_dist_m = X_dist_m[~np.isnan(X_dist_m).any(axis=1)] # Eliminação de linha com nan
    
        #% Deseho dos gráficos escolhidos
        '--------------------------------'
        #
        # for conta_med, conta_nome in zip(nome_dados_rota, medicoes_rota):
        for conta_med in medicoes_rota_01:
    
            # print(conta_med, conta_nome)
            
            dist_pot_rec = locals()[conta_med][var_ref_x].values
            dist_pot_rec = dist_pot_rec[~np.isnan(dist_pot_rec)]
        
            aten_pot_rec = locals()[conta_med][var_ref_y].values
            aten_pot_rec = aten_pot_rec[~np.isnan(aten_pot_rec)]
        
            plt.plot(dist_pot_rec, aten_pot_rec, label = conta_med)
            # plt.text(dist_pot_rec[-1], aten_pot_rec[-1],
            #          conta_med, c = 'b', fontsize  = 12)
        #
        plt.plot(X_dist_m, y_pred, lw = 2, ls = '--', c='r',
                 label = '%1.1f' %conta_alt_ant_tx + 'm \n' '%1.2e' %contf_freq_tx)
        #
        plt.title('Signal attenuation - rede: ') # + str(cam_ocultas))
        plt.xlabel('Distance - m')         # 'Distância - km'
        plt.ylabel('Signal attenuation - dB')    # 'Nível do sinal - dB'
        #
        plt.legend(bbox_to_anchor = (1, 1), loc='upper left',
                    fontsize = 8)
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
    X_pred = escala_keras.transform(X_predicao)
    y_pred = modelo_rna.predict(X_pred) 
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
#
#
'=============================================================================='
# %% Parte II - Análise de hiperparâmetros do modelo - Keras-tuner
# @todo
'=============================================================================='
#
# %% Rede neural artificial - arquitetura e modelo
'------------------------------------------------------------------------------'
#
'''
Ref.:
https://www.tensorflow.org/tutorials/keras/keras_tuner
'''
# Arquitetura da rede neural
'---------------------------'
def arquitetura_modelo_rna(hp): #, unidades=None, ativacao=None,
#                          retirada=None, taxa_aprendizado=None,
                           # num_entradas=None, num_cam=None, num_saidas=None):
    #
    num_entradas = 3
    num_camadas = 2
    num_saidas = 1
    
    # Especificação da variação dos hiperparâmetros
    '----------------------------------------------'
    # Ativação
    hp_ativacao = hp.Choice("activation", ["relu"])
    # Dropout
    hp_retirada = hp.Boolean("dropout")
    # Taxa de aprendizado
    hp_taxa_aprendizado = hp.Choice("taxa_aprend", values = [1e-2, 1e-3, 1e-4])
    # Número/quantidade de camadas
    hp_num_cam_1 = hp.Int("camada_1", min_value = 1, max_value = 9, step = 1)
    hp_num_cam_2 = hp.Int("camada_2", min_value = 1, max_value = 9, step = 1)
    #
    # Formação do tipo de modelo
    '---------------------------'
    modelo_rna = Sequential()
    #
    # Camada de entrada e camada(s) oculta(s)
    '----------------------------------------'
    conta_valor_arranjo = 0
    for cont_camada in range(num_camadas):
        print(cont_camada)
        #
        if conta_valor_arranjo == 0:
            #
            # Camada de entrada e 1a camada oculta
            '-------------------------------------'
            modelo_rna.add(Dense(units = hp_num_cam_1,
                                 input_shape = (num_entradas,),
                                 kernel_initializer = 'normal', # normal he_uniform
                                 activation = hp_ativacao))
            #
            # modelo_rna.add(Dense(units = hp_num_cam_1,
            #                      activation = hp_ativacao,
            #                      input_dim = num_entradas))
            #
            conta_valor_arranjo += 1
            #
        else:
            # 2a camadas ocultas em demais
            '-------------------------------------'
            modelo_rna.add(Dense(units = hp_num_cam_2,
                                 activation = hp_ativacao))
        #
    # Camada de saída
    '----------------'
    modelo_rna.add(Dense(units = num_saidas,
                          activation = 'linear'))
    #
    # Compilação do modelo
    '---------------------'
    modelo_rna.compile(optimizer = Adam(learning_rate = hp_taxa_aprendizado),
                       loss = 'mse',
                       metrics = ['mean_absolute_error'])
    #
    return modelo_rna
#
# %% Valores de entrada e saída e modelo da rede
'-----------------------------------------------'
#
# @todo: incluir num_entrada, num_camadas e num_saidas como parãmotrso criação
# da rede
#
num_entradas = len(var_entrada)
num_camadas = 2
num_saidas = len(var_saida)

# %% Ajustes de hiperparâmetros do modelo
'------------------------------------------------------------------------------'
#
# Biblioteca para ajuste do Keras
import keras_tuner as kt

hp = kt.HyperParameters()
#
# modelo_rna = construcao_modelo(hp) #, num_entradas, num_camadas, num_saidas)
modelo_rna = arquitetura_modelo_rna(hp)
#
# modelo_rna.summary()
# #
# plot_model(modelo_rna,
#            to_file="modelo_ANN.png",
#            show_shapes=True,
#            show_layer_names=True)
#
'----------------------------------------------------------------'
' =--> pode ir daqui para: Parte III - TensorBoard (linha 1899?)'
'----------------------------------------------------------------'
#
# %% Definição do algorítmo de busca / sintonia dos hiperparâmetros
'------------------------------------------------------------------'
#
# Ref.: 
# https://keras.io/api/keras_tuner/tuners/
# https://keras.io/guides/keras_tuner/getting_started/
#
# %% Tuner 
'---------------'
#
objetivo = 'val_loss' # val_loss 

# @todo:
''' 
# https://keras.io/api/keras_tuner/tuners/
# There are a few built-in Tuner subclasses available for widely-used tuning
# algorithms: RandomSearch, 
# BayesianOptimization and Hyperband.
'''
tuner = kt.RandomSearch(
    hypermodel = arquitetura_modelo_rna, # construcao_modelo(hp, num_entradas, num_camadas, num_saidas),
    objective = objetivo,
    max_trials = 15,
    executions_per_trial = 10,
    overwrite = True,
    directory="MedicoesProcessados",
    project_name="KerasTuner",
    seed = 42)

# tuner = kt.GridSearch(
#     hypermodel = arquitetura_modelo_rna, # construcao_modelo(hp, num_entradas, num_camadas, num_saidas),
#     objective = objetivo,
#     max_trials = 10,
#     executions_per_trial = 5,
#     overwrite = True,
#     directory="MedicoesProcessados",
#     project_name="KerasTuner",
#     seed = 42)
  
tuner.search_space_summary()

# %% Treinamento para busca pelos melhores hiperparâmetros
'---------------------------------------------------------'
#
es = EarlyStopping(monitor = 'loss', min_delta = 1e-5, # 1e-10
                    patience = 4, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2,
                        patience = 4, verbose = 1)
#
# Busca dos melhores hiperparêmtros
'----------------------------------'
tuner.search(X_train_escala, y_train,
             validation_data=(X_test_escala,y_test),
             batch_size = 100,
             epochs = 50,
             callbacks = [es, rlr])
#
# %% Avaliação dos melhores hiperparâmetros
'------------------------------------------'
#
# Lista do espaço de hiperparêmatros
'-----------------------------------'
tuner.search_space_summary(extended=False)

# Resultados encontrados
'-----------------------'
tuner.results_summary(1) # Melhor
tuner.results_summary(20) # Todos

# Melhores hiperparâmetros
'-------------------------'
n_modelo = 4
best_hps = tuner.get_best_hyperparameters(num_trials=n_modelo)[n_modelo-1]
# print(f"""
# The hyperparameter search is complete. 
# The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')}.
# """)
print(best_hps.get('activation'))
print(best_hps.get('taxa_aprend'))
print(best_hps.get('camada_1'))
print(best_hps.get('camada_2'))
# 
# %% Melhores modelos calculados em ordem de melhoria em termos da métrica
'-------------------------------------------------------------------------'
# @todo
# 'n_melhores' melhores modelos (em orde de melhor)
n_melhores = 4
melhor_modelo = tuner.get_best_models(num_models = n_melhores)[n_melhores-1]
# melhor_modelo = modelos[0]

# Construção do melhor modelo
# Needed for `Sequential` without specified `input_shape`.
melhor_modelo.build(input_shape=(None))
melhor_modelo.summary()
#
plot_model(melhor_modelo,
           to_file="modelo_ANN.png",
           show_shapes=True,
           show_layer_names=True)
#
# %% Construção do modelo com os melhores hiperparâmetros encontrados
'------------------------------------------------------------------------------'
#
# melhor_modelo = tuner.hypermodel.build(best_hps)
#
# %% Modelo com os melhores hiperparâmetros encontrados
'------------------------------------------------------------------------------'
#
# Pesos e vieses do melhor modelo
'--------------------------------'

melhor_modelo.get_weights()

print('Pesos e vieses das camadas do melhor modelo \n')
for layer in melhor_modelo.layers:
    print(layer.name)
    print("-------------------------------------------")
    print("Pesos - Forma: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
    print("--------------------------------------------")
    print("Viés - Forma: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')
#
# @todo
# for conta, layer in enumerate( melhor_modelo.layers):
#     print(conta)
#     print(layer)
#
# %% Arranjo da rede - melhor encontrada pelo keras-tuner
'--------------------------------------------------------'
#
melhor_modelo.summary()
#
arranjo_rede_val = ''
arranjo_rede = [] 
while True:
    '---------------------------------------------'
    lista = input('Entre com o arranjo da rede: ');
    if len(lista) == 0:
        break
    arranjo_rede_val = arranjo_rede_val + '_' + str(lista)
    arranjo_rede.append(int(lista))
#
# %% Gravação do modelo da rede neural
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

# ---------------------------------------
num_entradas = len(var_entrada)

caminho_1 = 'Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/'
caminho_2 = 'Programas/MedicoesProcessados/ModelosRNA/'

caminho_grava_modelo_rna = caminho_1 + caminho_2

nome_arq_rna = nome_arq + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '.h5'
#
# Gravação do arquivo
'--------------------'                
melhor_modelo.save(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))
#
# Gravação para processamento posterior e registro final da rede
'---------------------------------------------------------------'
data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")

nome_arq_rna = nome_arq  + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '_' + data_hora_gravacao + '.h5'
#
# Gravação do arquivo
'--------------------'                
melhor_modelo.save(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))
#
# %% Leitura do modelo escolhido da rede neural
'------------------------------------------------------------------------------'
#
del modelo_rna

modelo_rna_carregado = load_model(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))

modelo_rna = modelo_rna_carregado

modelo_rna.get_weights()

'----------------------------------------------------------------'
' =--> pode ir daqui para: Comparação com o modelo de referência log-distância com rotas'
'----------------------------------------------------------------'
#
# %% Treinamento do modelo com os melhores hiperparâmetros encontrados
'------------------------------------------------------------------------------'
#
# historico = melhor_modelo.fit(X_train_escala, y_train,
#                               epochs = 10,
#                               validation_split = 0.2,
#                               callbacks = [es, rlr])

historico = melhor_modelo.fit(x=X_train_escala, y=y_train,
              validation_data=(X_test_escala, y_test),
              batch_size = 100,
              epochs = 50,
              callbacks = [es, rlr])
#
val_loss_por_epoca = historico.history['val_loss']
print(val_loss_por_epoca)
#
import pandas as pd
pd.DataFrame(historico.history)
#
loss = historico.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#
'=============================================================================='
# %% Parte III - Análise utilizando Keras Tuner e TensorBoard
# @todo
'=============================================================================='
# 
'------------------------------------------------------------------------------'
# Ref.: https://keras.io/guides/keras_tuner/visualize_tuning/
#
# %% Tuner 
'---------------'
#
objetivo = 'val_loss' # val_loss val_accuracy

# @todo:
''' 
# https://keras.io/api/keras_tuner/tuners/
# There are a few built-in Tuner subclasses available for widely-used tuning
# algorithms: RandomSearch, 
# BayesianOptimization and Hyperband.
'''

# tuner = kt.RandomSearch(
#     hypermodel = arquitetura_modelo_rna, # construcao_modelo(hp, num_entradas, num_camadas, num_saidas),
#     objective = objetivo,
#     max_trials = 10,
#     executions_per_trial = 5,
#     overwrite = True,
#     directory="MedicoesProcessados",
#     project_name="KerasTuner",
#     seed = 42)

tuner = kt.GridSearch(
    hypermodel = arquitetura_modelo_rna, # construcao_modelo(hp, num_entradas, num_camadas, num_saidas),
    objective = objetivo,
    max_trials = 10,
    executions_per_trial = 5,
    overwrite = True,
    directory="MedicoesProcessados",
    project_name="KerasTuner",
    seed = 42)

tuner.search_space_summary()
#
# %% Tensorboard - configuração
'------------------------------' 
#
num_entradas = len(var_entrada)
num_saidas = len(var_saida)

caminho_1 = r'Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao'
caminho_2 = '\Artigo_03_Vegetacao\Programas\MedicoesProcessados'
caminho_3 = '\DirLogTensorBoard'
#
# caminho_log = caminho_1 + caminho_2 + caminho_3
# data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm")
data_hora_gravacao = time.strftime("%d%b%y_%Hh%Mm")
#
nome_arq_rna = '\Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '_' + data_hora_gravacao # + '.sav'
#
caminho_log = caminho_1 + caminho_2 + caminho_3 #  + nome_arq_rna
#
tensorboard_cb = TensorBoard(log_dir = caminho_log, histogram_freq = 1)
#
# %% Treinamento para busca pelos melhores hiperparâmetros
'---------------------------------------------------------'
#
es = EarlyStopping(monitor = 'loss', min_delta = 1e-6,
                    patience = 2, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2,
                        patience = 2, verbose = 1)
#
# Busca dos melhores hiperparêmtros
'----------------------------------'
tuner.search(X_train_escala, y_train,
             validation_data=(X_test_escala,y_test),
             batch_size = 100,
             epochs = 50,
             callbacks = [es, rlr, tensorboard_cb])
#
# %% Avaliação dos hiperparâmetros
'---------------------------------'
#
# Lista do espaço de hiperparêmatros
'-----------------------------------'
tuner.search_space_summary(extended=False)

# Resultados encontrados
'-----------------------'
tuner.results_summary(1) # Melhor
tuner.results_summary() # Todos

# Melhores hiperparâmetros
'-------------------------'
best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]
# print(f"""
# The hyperparameter search is complete. 
# The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')}.
# """)
print(best_hps.get('activation'))
print(best_hps.get('taxa_aprend'))
print(best_hps.get('camada_1'))
print(best_hps.get('camada_2'))
#
'--------------------------------------------------------------------'
' =--> pode ir daqui para: Inicialização do Tensorboard (linha 2016?)'
'--------------------------------------------------------------------'
#
# %% Melhores modelos calculados em ordem de melhoria em termos da métrica
'-------------------------------------------------------------------------'
# @todo
# 'n_melhores' melhores modelos (em orde de melhor)
n_melhores = 1
modelos = tuner.get_best_models(num_models = n_melhores)
melhor_modelo = modelos[0]

# Construção do melhor modelo
# Needed for `Sequential` without specified `input_shape`.
melhor_modelo.build(input_shape=(None))
melhor_modelo.summary()
#
plot_model(melhor_modelo,
           to_file="modelo_ANN.png",
           show_shapes=True,
           show_layer_names=True)
#
# %% Construção do modelo com os melhores hiperparâmetros encontrados
'------------------------------------------------------------------------------'

# melhor_modelo = tuner.hypermodel.build(best_hps)
#
# %% Modelo com os melhores hiperparâmetros encontrados
'------------------------------------------------------------------------------'
#
# Pesos e vieses do melhor modelo
'--------------------------------'

melhor_modelo.get_weights()

print('Pesos e vieses das camadas do melhor modelo \n')
for layer in melhor_modelo.layers:
    print(layer.name)
    print("-------------------------------------------")
    print("Pesos - Forma: ",layer.get_weights()[0].shape,'\n',layer.get_weights()[0])
    print("--------------------------------------------")
    print("Viés - Forma: ",layer.get_weights()[1].shape,'\n',layer.get_weights()[1],'\n')
#
# @todo
# for conta, layer in enumerate( melhor_modelo.layers):
#     print(conta)
#     print(layer)
#
# %% Treinamento do modelo com os melhores hiperparâmetros encontrados
'------------------------------------------------------------------------------'
#
# historico = melhor_modelo.fit(X_train_escala, y_train,
#                               epochs = 10,
#                               validation_split = 0.2,
#                               callbacks = [es, rlr])

historico = melhor_modelo.fit(x=X_train_escala, y=y_train,
              validation_data=(X_test_escala, y_test),
              batch_size = 100,
              epochs = 50,
              callbacks = [es, rlr])
#
val_loss_por_epoca = historico.history['val_loss']
print(val_loss_por_epoca)
#
import pandas as pd
pd.DataFrame(historico.history)
#
loss = historico.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#
# %% Arranjo da rede - melhor encontrada pelo keras-tuner
'--------------------------------------------------------'
#
arranjo_rede_val = ''
arranjo_rede = [] 
while True:
    '---------------------------------------------'
    lista = input('Entre com o arranjo da rede : ');
    if len(lista) == 0:
        break
    arranjo_rede_val = arranjo_rede_val + '_' + str(lista)
    arranjo_rede.append(int(lista))
#
# %% Gravação do modelo da rede neural
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

# ---------------------------------------
num_entradas = len(var_entrada)

caminho_1 = 'Y:/3_Arquivos/10_Doutorado/UFF/a_ArtigosElaboracao/Artigo_03_Vegetacao/'
caminho_2 = 'Programas/MedicoesProcessados/ModelosRNA/'

caminho_grava_modelo_rna = caminho_1 + caminho_2

nome_arq_rna = nome_arq + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '.h5'
#
# Gravação do arquivo
'--------------------'                
melhor_modelo.save(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))
#
# Gravação para processamento posterior e registro final da rede
'---------------------------------------------------------------'
data_hora_gravacao = time.strftime("data_%d%b%y_%Hh%Mm%Ss")

nome_arq_rna = nome_arq  + '_Rede_' + \
                str(num_entradas) + '_' + \
                ''.join(str(x) for x in arranjo_rede) + '_' + \
                str(num_saidas) + '_' + data_hora_gravacao + '.h5'
#
# Gravação do arquivo
'--------------------'                
melhor_modelo.save(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))
#
# %% Leitura do modelo escolhido da rede neural
'------------------------------------------------------------------------------'
#
del modelo_rna

modelo_rna_carregado = load_model(os.path.join(caminho_grava_modelo_rna, nome_arq_rna))

modelo_rna = modelo_rna_carregado

'----------------------------------------------------------------'
' =--> pode ir daqui para: Comparação com o modelo de referência log-distância com rotas'
'----------------------------------------------------------------'
#
#
#
'------------------------------------------------------------------------------'
# %% Inicialização do Tensorboard
'=============================================================================='
#
import os
from threading import Thread
#
# funções
'-----------------------'
def tarefa_sistema(caminho_log):
    porta = 6006
    print ('\nIniciando tensor board')
    # comando = r'tensorboard --logdir Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao\Artigo_03_Vegetacao\Programas\MedicoesProcessados\KerasTuner --port ' + str(porta)
    comando = r'tensorboard --logdir Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao\Artigo_03_Vegetacao\Programas\MedicoesProcessados\DirLogTensorBoard --port ' + str(porta)
    # comando = r'tensorboard --logdir ' + caminho_log + ' --port ' + str(porta)
    comando = comando.replace('\\', '/')
    # print(comando)
    os.system(comando)
    print ('\n Tensor board finalizado')
#
os.system('npx kill-port 6006')

# Carregar tensorBoard
'---------------------'
# %load_ext tensorboard
# %reload_ext tensorboard
os.system('reload_ext tensorboard')


'-----------------------------------------------------'
# Cria Thread (tarefa)
tarefa_1 = Thread(target = tarefa_sistema, args=(caminho_log,))

# Inicia Thread
tarefa_1.start()

'-----------------------------------------------------'
import webbrowser
porta = 6006
url_tb = 'http://localhost:' + str(porta)

# Executa TansorBoard no Navegador Padrão
'----------------------------------------'
webbrowser.open(url_tb)


# %% Finaliza Tread ao finalizar o estudo
'----------------------------------------'
#
tarefa_1.join()















# %%
#
'----------------------------------'
# Rascunho e testa daqui par abaixo
'----------------------------------'
#
# %%
# FIXME
# @todo
# %% Tensorboard
'---------------' 

#
dir_log_tensorBoard = 'dir_log_tensorBoard'

raiz_dir_log = os.path.join(os.curdir, 'dir_log_tensorBoard')
#
def get_run_logdir():
    #
    run_id = time.strftime("rodada_%d_%m_%Y-%H_%M_%S")
    #
    return os.path.join(raiz_dir_log, run_id)
#
raiz_dir_log = get_run_logdir()

tensorboard_cb = TensorBoard(log_dir = raiz_dir_log)

# melhor_modelo
# modelo_rna


historico = modelo_rna.fit(x=X_train_escala, y=y_train,
              validation_data=(X_test_escala, y_test),
              batch_size = 100,
              epochs = 50,
              callbacks = [tensorboard_cb])
#
# %%

historico = modelo_rna.fit(x=X_test_escala,y=y_test,
                              validation_data=(X_test,y_test),
                              batch_size = 48,
                              epochs = 10,
                              callbacks = [tensorboard_cb])

# %% Gráficos da perde
'------------------------------------------------------------------------' 
# Ref.: https://www.youtube.com/watch?v=PG4XGqUeYnM
#
loss = historico.history['loss']
val_loss = historico.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#
# %%
'------------------------------------------------------------------------' 
#
acc = historico.history['accuracy']
val_acc = historico.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% Carregando o TensorBoard
'------------------------------------------------------------------------------'
#
import os
import IPython
'--------------------------------'
# comando = '%load_ext tensorboard'
# os.system(comando)
%load_ext tensorboard

%reload_ext tensorboard


'--------------------------------'
comando_1= '%tensorboard'
argumento_1 = '--logdir=\''
comando = comando_1 + ' ' + argumento_1

caminho_1 = r'Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao'
caminho_2 = '\Artigo_03_Vegetacao\Programas\MedicoesProcessados'
caminho_3 = '\dir_log_tensorBoard\''
caminho = caminho_1 + caminho_2 + caminho_3

argumento_2 = '--port=6007'

%tensorboard --logdir=caminho

comando_final = comando + caminho + ' ' + argumento_2

print(comando_final)

# run(comando_final)   
# os.system(comando_final)

%tensorboard --logdir='Y:\\3_Arquivos\\10_Doutorado\\UFF\\a_ArtigosElaboracao\\Artigo_03_Vegetacao\\Programas\\MedicoesProcessados\\dir_log_tensorBoard' --port=6007

# Exemplo:
# %tensorboard --logdir='Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao\Artigo_03_Vegetacao\Programas\TensorBoard_log_dir' --port=6007


# 

# http://localhost:6006

# Ou no sistema, no diretório onde estão os 'log'
# python -m %tensorboard --logdir='Y:\3_Arquivos\10_Doutorado\UFF\a_ArtigosElaboracao\Artigo_03_Vegetacao\Programas\TensorBoard_log_dir\'


#%%

export_path = os.path.join('/my_dir/', 'teste')
melhor_modelo.save(export_path)
# models.save_model(melhor_modelo, export_path)
# http://localhost:6006














