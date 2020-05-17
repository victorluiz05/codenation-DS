#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[51]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[52]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[12]:


# Sua análise da parte 1 começa aqui.


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[53]:


def q1():
    q1_norm = sct.scoreatpercentile(dataframe['normal'], 25)
    q2_norm = sct.scoreatpercentile(dataframe['normal'], 50)
    q3_norm = sct.scoreatpercentile(dataframe['normal'], 75)
    q1_binom = sct.scoreatpercentile(dataframe['binomial'], 25)
    q2_binom = sct.scoreatpercentile(dataframe['binomial'], 50)
    q3_binom = sct.scoreatpercentile(dataframe['binomial'], 75)
    resposta_q1 = (round(q1_norm - q1_binom, 3) , round(q2_norm - q2_binom, 3), round(q3_norm-q3_binom, 3))
    return resposta_q1


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[108]:


def q2():
    df_normal = dataframe['normal']
    media = df_normal.mean()
    desvio = df_normal.std()
    intervalo = df_normal[(df_normal>=(media-desvio)) & (df_normal<=(media+desvio))]
    resposta_q3 = intervalo.count()/len(df_normal)
    return float(round(resposta_q3, 3))


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[110]:


def q3():
    df_normal = dataframe['normal']
    df_binomial = dataframe['binomial']
    resposta_q3 = (df_binomial.mean() - df_normal.mean(), df_binomial.var() - df_normal.var())
    return tuple(np.round(resposta_q3, 3))


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[58]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[59]:


# Sua análise da parte 2 começa aqui.


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[105]:


def q4():
    false_pulsar_mean_profile = stars.mean_profile[stars.target == 0]
    media = false_pulsar_mean_profile.mean()
    std = false_pulsar_mean_profile.std()
    false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - media) / std
    quantis = sct.norm.ppf([0.80, 0.90, 0.95])
    cdf_model = ECDF(false_pulsar_mean_profile_standardized)
    resposta_q4 = cdf_model(quantis)
    return tuple(np.round(resposta_q4, 3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[112]:


def q5():
    false_pulsar_mean_profile = stars.mean_profile[stars.target == 0]
    media = false_pulsar_mean_profile.mean()
    std = false_pulsar_mean_profile.std()
    false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - media) / std
    
    quantis_teoricos = sct.norm.ppf([0.25, 0.50, 0.75])
    
    Q1 = sct.scoreatpercentile(false_pulsar_mean_profile_standardized, 25)
    Q2 = sct.scoreatpercentile(false_pulsar_mean_profile_standardized, 50)
    Q3 = sct.scoreatpercentile(false_pulsar_mean_profile_standardized, 75)
    resposta_q5 = (Q1 - quantis_teoricos[0], Q2 - quantis_teoricos[1], Q3 - quantis_teoricos[2])
    return tuple(np.round(resposta_q5, 3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
