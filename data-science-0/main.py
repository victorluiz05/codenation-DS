#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


black_friday = pd.read_csv("black_friday.csv")


# In[4]:


## Inicie sua análise a partir daqui


# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[8]:


def q1():
    questao1 = black_friday.shape
    return questao1


# In[ ]:


## Questão 2

#Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.


# In[5]:


def q2():
    n_mulheres = len(black_friday[black_friday['Age'] == '26-35'].loc[black_friday['Gender'] == 'F'])
    return n_mulheres


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    unicos = len(black_friday['User_ID'].unique())
    return unicos


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    unique_dtype = len(black_friday.dtypes.unique())
    return unique_dtype


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    porcentagem_nulos = (len(black_friday) - len(black_friday.dropna())) / len(black_friday)
    return porcentagem_nulos


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[12]:


def q6():
    max_col_null = black_friday.isnull().sum().max()
    return int(max_col_null)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    moda_p3 = black_friday['Product_Category_3'].mode()
    return float(moda_p3)


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    valor_max = black_friday['Purchase'].max()
    valor_min = black_friday['Purchase'].min()
    black_friday_norm = (black_friday['Purchase'] - valor_min)/(valor_max - valor_min)
    media_normalizada = black_friday_norm.mean()
    return float(media_normalizada)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[9]:


def q9():
    media = black_friday['Purchase'].mean()
    desvio_padrao = black_friday['Purchase'].std()
    z_score = (black_friday['Purchase'] - media) / desvio_padrao
    n_ocorrencias = z_score.apply(lambda x: True if(1 >= x >= -1) else False).sum()
    return int(n_ocorrencias)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[10]:


def q10():
    categoria2 = black_friday[black_friday['Product_Category_2'].isnull()].index
    categoria3 = black_friday[black_friday['Product_Category_3'].isnull()].index
    for categoria2, categoria3 in zip(categoria2,categoria3):
        if categoria2 == categoria3:
            return  True
        else: 
            return False

