#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:55:42 2021

@author: yichizhang
"""

import numpy as np
from random import choices
from random import sample
import math
import matplotlib.pyplot as plt

#%%
"""
Model setup
"""

class Crowdsourcing:
    def __init__(self, w, Gamma_w, Gamma_s, 
                       e = 1,
                       m = 100, 
                       n0 = 10, 
                       mi = 30, 
                       prior_e = [0.2, 0.8, 0], 
                       signal = [1,2,3,4,5]):
        self.m = m # Total number of tasks
        self.n0 = n0 # Minimum number of agents that are assigned to each task
        self.mi = mi # Number of tasks that each agent answers
        self.prior_e = prior_e # Prior of three types of agents: shirker, rational worker, honest worker
        self.signal = signal # Signal space
        self.S = len(signal)
        self.w = w # Prior of ground truth
        self.Gamma_w = Gamma_w #Confusion matrix of full effort worker
        self.Gamma_s = Gamma_s #Confusion matrix of shirker
        self.e = e #Effort level of rational agents

def Report_Generator(para):
    m = para.m
    n0 = para.n0
    mi = para.mi
    prior_e = para.prior_e
    w = para.w
    e = para.e
    Gamma_w = para.Gamma_w
    Gamma_s = para.Gamma_s
    signal = para.signal
    n = math.ceil(m*n0/(mi - 1))
    
    agent = np.arange(len(prior_e))
    agent_e = choices(agent, prior_e, k = n)
    Y = choices(list(range(1, len(w)+1)), w, k = m)
    
    flag = 0
    while flag < m:
        flag = 0
        current_task = np.zeros(n)
        R = np.zeros((n, m))
        for j, y in enumerate(Y):
            if len(list(np.where(current_task < mi)[0])) < n0:
                break
            else:
                flag += 1
            random_agents = np.array(sample(list(np.where(current_task < mi)[0]), n0))
            for i, index in enumerate(random_agents):
                if agent_e[index] == 0:
                    Gamma_i = Gamma_s
                elif agent_e[index] == 1:
                    Gamma_i = e*Gamma_w + (1-e)*Gamma_s
                elif agent_e[index] == 2:
                    Gamma_i = Gamma_w
                R[index,j] = choices(signal, Gamma_i[Y[j]-1])[0]
            current_task[random_agents] += 1
    
    agent_i = np.where(np.count_nonzero(R, axis = 1) < mi)[0]
    for i in agent_i:
        if agent_e[index] == 0:
            Gamma_i = Gamma_s
        elif agent_e[index] == 1:
            Gamma_i = e*Gamma_w + (1-e)*Gamma_s
        elif agent_e[index] == 2:
            Gamma_i = Gamma_w
        more_tasks = sample(list(np.where(R[i] == 0)[0]), mi - np.count_nonzero(R[i]))
        for j in more_tasks:
            R[i,j] = choices(signal, Gamma_i[Y[j]-1])[0]

    
    return (R, Y, np.array(agent_e))


#%%
"""
Pairing mechanism
"""
def Soft_pred_to_MI(Marginal, pred, b, p, q, Y_T, f):
    # Mi = np.zeros(5)
    # K_b = np.zeros(5)
    # K_p = np.zeros(5)
    Mi = 0
    K_b = 0
    K_p = 0
    predict_b = pred[b]
    predict_p = pred[p]

    if Marginal[Y_T[b]-1] != 0:
        Jp_b = predict_b[Y_T[b]-1]/Marginal[Y_T[b]-1]
        if f == 0:
            if Jp_b > 1:
                K_b = 0.5
            elif Jp_b < 1:
                K_b = -0.5
        elif f == 1:
            if 1 + np.log(Jp_b) < -10:
                K_b = -10
            elif 1 + np.log(Jp_b) > 10:
                K_b = 10
            else:
                K_b = 1 + np.log(Jp_b)
        elif f == 2:
            K_b = 2*(Jp_b - 1)
        elif f == 3:
            if 1 - 1/np.sqrt(Jp_b) < -10:
                K_b = -10
            else:
                K_b = 1 - 1/np.sqrt(Jp_b)
        elif f == 4:
            K_b = 4*np.power(Jp_b - 1, 3)
        elif f == 5:
            K_b = 6*np.power(Jp_b - 1, 5)
        
    if Marginal[Y_T[q]-1] != 0:
        Jp_p = predict_p[Y_T[q]-1]/Marginal[Y_T[q]-1]
        if f == 0:
            if Jp_p > 1:
                K_p = 0.5
            elif Jp_p < 1:
                K_p = -0.5
            Mi = K_b - K_p
            
        elif f == 1:
            kl = 1 + np.log(Jp_p)
            if kl < -10:
                K_p = -10
            elif kl > 10:
                K_p = 10
            else:
                K_p = kl
            Mi = K_b - np.exp(K_p-1)
            
        elif f == 2:
            K_p = 2*(Jp_p - 1)
            Mi = K_b - np.square(K_p)/4 - K_p
            
        elif f == 3:
            heli = 1 - 1/np.sqrt(Jp_p)
            if heli < -10:
                K_p = -10
            else:
                K_p = heli
            Mi = K_b - K_p/(1-K_p)
            
        elif f == 4:
            K_p = 4*np.power(Jp_p - 1, 3)
            Mi = K_b - np.power(np.abs(K_p)/4, 1/3)*np.abs(K_p)*3/4 - K_p
        elif f == 5:
            K_p = 6*np.power(Jp_p - 1, 5)
            Mi = K_b - np.power(np.abs(K_p)/6, 1/5)*np.abs(K_p)*5/6 - K_p
    
    return Mi



def soft_predictor_learner(X, para):
    S = para.S
    X = np.array(X)
    m = np.size(X, axis = 1)
    P = []
    for j in range(m):
        pj =  np.zeros(S)
        nj = np.count_nonzero(X[:,j])
        if nj != 0:
            for i in range(S):
                pj[i] = np.count_nonzero(X[:,j] == i+1)/nj     
        P.append(pj)
    return np.array(P)

def LR_learner(X, Y, f, para):
    S = para.S
    signal = para.signal
    Y = np.array(Y)
    T = np.size(np.where(Y != 0)[0])*10
    index_i = np.where(Y != 0)[0]
    Yi = Y[index_i]
    M = 0
    
    Y_int = Y.copy().astype(int)
    pred = soft_predictor_learner(X, para)
    Marginal = np.zeros(S)
    for j in signal:
        Marginal[j-1] = np.count_nonzero(Yi == j)/len(Yi)
    for j in range(T):
        b,q = sample(list(index_i), 2)
        p = sample(range(len(Y)), 1)[0]
        while p == b or p == q:
            p = sample(range(len(Y)),1)[0]
        Mi = Soft_pred_to_MI(Marginal, pred, b, p, q, Y_int, f)
        M += Mi
    M = M/(T)

    return M


# Pairing mechanism, can be used as black box. Now it requires ground truth, which is a perfect predictor of agents' reports.
# In reality, Gr can be replaced by a soft learning algorithm.
def mechanism_pairing(X, f, para):
    n = len(X)

    X = np.array(X)
    U = np.zeros(n)
    for i in range(n):
        X_ni = np.vstack((X[0:i], X[i+1:n]))
        U[i] = LR_learner(X_ni, X[i], f, para)

    return U

#%%
"""
Matrix mechanism
"""


def distribution_estimator(X1, X2, para): #learn empirical distributions from two agents' reports
    S = para.S
    X1 = np.array(X1)
    X2 = np.array(X2)
    P = np.zeros((S,S))
    Q1 = np.zeros(S)
    Q2 = np.zeros(S)
    m = len(X1)
    m1 = np.count_nonzero(X1)
    m2 = np.count_nonzero(X2)
    m12 = np.size(np.intersect1d(np.where(X1 != 0), np.where(X2 != 0)))
    
    for i in range(m):
        if X1[i] != 0 and X2[i] != 0:
            P[int(X1[i])-1][int(X2[i])-1] += 1/m12
        if X1[i] != 0:
            Q1[int(X1[i])-1] += 1/m1
        if X2[i] != 0:
            Q2[int(X2[i])-1] += 1/m2
    
    return P, Q1, Q2

def distribution_learner(report, predictor, para): #another version of distribution_estimator, which takes a report vector and a soft predictor as inputs
    S = para.S
    r = report
    index_i = np.where(r != 0)[0]
    pre = predictor
    P = np.zeros((S,S))
    for j in index_i:
        P[int(r[j])-1]+=pre[j]/len(index_i)
    Q_a = np.zeros(S)
    for j in range(S):
        Q_a[j] = np.count_nonzero(r == j+1)/len(index_i)
    #Q_p = np.sum(pre, axis = 0)/len(pre)
    #Q = Q_p*Q_a.reshape(-1,1)
    return P, Q_a

def MI_computer(P, Q1, Q2, f): #For matrix mechanism, learn different types of mutual information
    Q = Q2*Q1.reshape(-1, 1)
    
    if f == 0:
        MI = np.sum(np.absolute(P - Q))
    elif f == 1:
        t = P*np.log(P/Q)
        nan_ind = np.isnan(t)
        t[nan_ind] = 0 
        MI = np.sum(t)
    elif f == 2:
        t = P*(np.square((Q/P)-1))
        nan_ind = np.isnan(t)
        t[nan_ind] = 0 
        MI = np.sum(t)
    elif f == 3:
        t = P*(np.square(np.sqrt(Q/P)-1))
        nan_ind = np.isnan(t)
        t[nan_ind] = 0 
        MI = np.sum(t)
    elif f == 4:
        t = P*(np.power((Q/P)-1, 4))
        nan_ind = np.isnan(t)
        t[nan_ind] = 0 
        MI = np.sum(t)
    elif f == 5:
        t = P*(np.power((Q/P)-1, 6))
        nan_ind = np.isnan(t)
        t[nan_ind] = 0 
        MI = np.sum(t)
    
    return MI

def mechanism_matrix_fast(X, f, para):
    m = para.m
    X = np.array(X)
    n = np.size(X, axis = 0)
    U = np.zeros(n)
    for i in range(n):
        X_ni = np.vstack((X[0:i], X[i+1:n]))
        Pi = soft_predictor_learner(X_ni, para)
        Q_p = np.sum(Pi, axis = 0)/m
        P, Q_a = distribution_learner(X[i], Pi, para)
        U[i] = MI_computer(P, Q_a, Q_p, f)
    return U

def mechanism_matrix(X):
    X = np.array(X)
    n = np.size(X, axis = 0)
    U = np.zeros((n,4))
    for i in range(n):
        lis = list(range(n))
        lis.remove(i)
        for j in lis:
            P, Q1, Q2 = distribution_estimator(X[i], X[j], para)
            U[i] += MI_computer(P, Q1, Q2)/(n-1)
    return U

#%%
"""
Determinant mechanism
"""
def distribution_learner_DMI(report, predictor, para):
    S = para.S
    r = report
    index_i = np.where(r != 0)[0]
    pre = predictor
    P = np.zeros((S,S))
    for j in index_i:
        P[int(r[j])-1]+=pre[j]/len(index_i)
    return P

def mechanism_determinant_fast(X, para):
    n = np.size(X, axis = 0)
    mi = np.count_nonzero(X[0])
    U = np.zeros(n)
    for i in range(n):
        X_ni = np.vstack((X[0:i], X[i+1:n]))
        Pi = soft_predictor_learner(X_ni, para)
        Xi_1 = X[i].copy()
        Xi_1[np.where(X[i] != 0)[0][0:int(mi/2)]] = 0
        Xi_2 = X[i].copy()
        Xi_2[np.where(X[i] != 0)[0][int(mi/2):mi]] = 0
        
        P1 = distribution_learner_DMI(Xi_1, Pi, para)
        P2 = distribution_learner_DMI(Xi_2, Pi, para)
        
        U[i] = np.linalg.det(P1)*np.linalg.det(P2)
    return U

def determinant_computer(X1,X2, para):
    S = para.S
    index = np.intersect1d(np.where(X1 != 0), np.where(X2 != 0))
    m12 = len(index)
    P1 = np.zeros((S,S))
    for j in index[0:int(m12/2)]:
        P1[int(X1[j])-1][int(X2[j])-1] += 1/int(m12/2)
    P2 = np.zeros((S,S))
    for j in index[int(m12/2):m12]:
        P1[int(X1[j])-1][int(X2[j])-1] += 1/(m12-int(m12/2))
    return np.linalg.det(P1)*np.linalg.det(P2)

def mechanism_determinant(X):
    n = np.size(X, axis = 0)
    U = np.zeros(n)
    for i in range(n):
        lis = list(range(n))
        lis.remove(i)
        for j in lis:    
            U[i] += determinant_computer(X[i],X[j], para)/(n-1)
    return U

#%%
"""
Plot agents mutual information scores and vs threshold
"""
w = np.array([0.2, 0.25, 0.25, 0.15, 0.15])
Gamma = np.array([[0.8, 0.1, 0.04, 0.04, 0.02], 
          [0.1, 0.8, 0.08, 0, 0.02],
          [0.01, 0.08, 0.82, 0.06, 0.03],
          [0, 0, 0.08, 0.85, 0.07],
          [0, 0.01, 0.01, 0.08, 0.9]])
Gamma_shirking = np.array([[0.26, 0.34, 0.26, 0.08, 0.06], 
                            [0.24, 0.3, 0.3, 0.11, 0.05],
                            [0.1, 0.28, 0.24, 0.26, 0.12],
                            [0.06, 0.13, 0.28, 0.3, 0.23],
                            [0.12, 0.08, 0.22, 0.3, 0.28]])
Gamma_random = np.ones((5,5))/5

para = Crowdsourcing(w = w, Gamma_w = Gamma, Gamma_s = Gamma_shirking, e = 0.78, m = 200, mi = 20, n0 = 8)
R, Y, agent_e = Report_Generator(para)
para_s = Crowdsourcing(w = w, Gamma_w = Gamma, Gamma_s = Gamma_random, prior_e = [1,0,0], m = 200, mi = 20, n0 = 2)
R_s, _, _ = Report_Generator(para_s)
P_all = mechanism_determinant_fast(np.vstack((R, R_s)), para)
n_w = np.size(R, axis = 0)
n_s = np.size(R_s, axis = 0)
P = P_all[0:n_w]
t_vs = np.max(P_all[n_w:n_w+n_s])

plt.figure()
plt.scatter(P[agent_e == 0], np.random.uniform(0,1,len(P[agent_e == 0])), color = 'red')
plt.scatter(P[agent_e == 1], np.random.uniform(0,1,len(P[agent_e == 1])), color = 'green')
plt.axvline(x = t_vs)
plt.title('Matrix_HLG')
#%%
print(R)



