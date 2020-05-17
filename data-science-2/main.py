#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as sct
import seaborn as sns


# In[43]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[5]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.
athletes.columns


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[29]:


def q1():
    height_sample = get_sample(athletes, 'height', 3000)
    shapiro_test = sct.shapiro(height_sample)
    if shapiro_test[1] < 0.05:
        resposta_q1 = False
    else:
        resposta_q1 = True
    return resposta_q1


# In[47]:


height_sample = get_sample(athletes, 'height', 3000)
sns.distplot(height_sample, bins = 25)
sm.qqplot(height_sample, fit=True, line="45")


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[34]:


def q2():
    height_sample = get_sample(athletes, 'height', 3000)
    jarque_test = sct.jarque_bera(height_sample)
    if jarque_test[1] < 0.05:
        resposta_q2 = False
    else:
        resposta_q2 = True
    return resposta_q2


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[48]:


def q3():
    weight_sample = get_sample(athletes, 'weight', 3000)
    dagostino_test = sct.stats.normaltest(weight_sample)
    if dagostino_test[1] < 0.05:
        resposta_q3 = False
    else:
        resposta_q3 = True
    return resposta_q3


# In[49]:


weight_sample = get_sample(athletes, 'weight', 3000)
sns.distplot(weight_sample)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[59]:


def q4():
    weight_sample = get_sample(athletes, 'weight', 3000)
    weight_sample_log = np.log(weight_sample)
    dagostino_test = sct.stats.normaltest(weight_sample_log)
    if dagostino_test[1] < 0.05:
        resposta_q4 = False
    else:
        resposta_q4 = True
    return resposta_q4


# In[58]:


weight_sample = np.log(get_sample(athletes, 'weight', 3000))
sns.distplot(weight_sample, bins=25)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[100]:


def q5():
    bra = athletes[athletes.nationality == 'BRA']
    usa = athletes[athletes.nationality == 'USA']
    ind_test = sct.ttest_ind(bra['height'], usa['height'], equal_var=False,nan_policy='omit')
    if ind_test[1] < 0.05:
        resposta_q5 = False
    else:
        resposta_q5 = True
    return resposta_q5


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[101]:


def q6():
    bra = athletes[athletes.nationality == 'BRA']
    can = athletes[athletes.nationality == 'CAN']
    ind_test = sct.ttest_ind(bra['height'], can['height'], equal_var=False,nan_policy='omit')
    if ind_test[1] < 0.05:
        resposta_q6 = False
    else:
        resposta_q6 = True
    return resposta_q6


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[103]:


def q7():
    usa = athletes[athletes.nationality == 'USA']
    can = athletes[athletes.nationality == 'CAN']
    #usa_sample = get_sample(usa, 'height', 320)
    #can_sample = get_sample(can, 'height', 320)
    ind_test = sct.ttest_ind(usa['height'], can['height'], equal_var=False,nan_policy='omit')
    return float(np.round(ind_test[1], 8))


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
