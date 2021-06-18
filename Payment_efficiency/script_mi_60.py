#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:52:55 2021

@author: yichizhang
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from random import choices
from random import sample
import warnings
warnings.filterwarnings('ignore')


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


def Report_Generator_2(para, E, n_E):
    m = para.m
    n0 = para.n0
    mi = para.mi
    w = para.w
    e = para.e
    Gamma_w = para.Gamma_w
    Gamma_s = para.Gamma_s
    signal = para.signal
    n = np.sum(n_E)
    
    agent = np.arange(len(E))
    agent_e = np.zeros(n)
    for i in range(len(n_E)-1):
        agent_e[np.sum(np.array(n_E)[0:i+1]):np.sum(np.array(n_E)[0:i+2])] = i+1
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
                e = E[int(agent_e[index])]
                Gamma_i = e*Gamma_w + (1-e)*Gamma_s
                R[index,j] = choices(signal, Gamma_i[Y[j]-1])[0]
            current_task[random_agents] += 1
    
    agent_i = np.where(np.count_nonzero(R, axis = 1) < mi)[0]
    for i, index in enumerate(agent_i):
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

def mechanism_matrix(X,para):
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

def mechanism_matrix_gt(X, f, x, para):
    n = np.size(X, axis = 0)
    U = np.zeros(n)
    for i in range(n):
        P, Q1, Q2 = distribution_estimator(X[i], x, para)
        U[i] = MI_computer(P, Q1, Q2, f)
    return U

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

def mechanism_determinant(X,para):
    n = np.size(X, axis = 0)
    U = np.zeros(n)
    for i in range(n):
        lis = list(range(n))
        lis.remove(i)
        for j in lis:    
            U[i] += determinant_computer(X[i],X[j], para)/(n-1)
    return U

def mechanism_determinant_gt(X, x, para):
    n = np.size(X, axis = 0)
    U = np.zeros(n)
    for i in range(n):
        U[i] = determinant_computer(X[i], x, para)
    return U

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


# def soft_predictor_learner(X):
#     X = np.array(X)
#     m = np.size(X, axis = 1)
#     P = []
#     for j in range(m):
#         pj =  np.zeros(S)
#         nj = np.count_nonzero(X[:,j])
#         if nj != 0:
#             for i in range(S):
#                 pj[i] = np.count_nonzero(X[:,j] == i+1)/nj     
#         P.append(pj)
#     return np.array(P)

def LR_learner(X, Y, f, para):
    Y = np.array(Y)
    T = np.size(np.where(Y != 0)[0])*5
    index_i = np.where(Y != 0)[0]
    Yi = Y[index_i]
    M = 0
    S = para.S
    
    Y_int = Y.copy().astype(int)
    pred = soft_predictor_learner(X,para)
    Marginal = np.zeros(S)
    for j in range(1,S+1):
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

"""
Model functions
"""
def cost(x, a):
    b = 1/(1/(1+np.exp(-a))-1/(1+np.exp(-0.1*a)))
    c = 1 - b/(1+np.exp(-a))
    return np.log((x-c)/(b+c-x))/a

def cost_2(x, a):
    return a*np.log((x+1)/(1-x)) + 0.1

def cost_2_inverse(c, a):
    return (np.exp((c-0.1)/a)-1)/(np.exp((c-0.1)/a)+1)

def cost_2_derivative(x, a):
    return 2*a/(1-np.power(x,2))

def accuracy_computer_threshold(R, Y, agent_w):
    R = np.array(R)
    R_w = R[np.where(agent_w == 1)[0]]
    m = np.size(Y)
    Y_tilde = np.zeros(m)
    for j in range(m):
        Rj = R_w[:,j]
        if np.count_nonzero(Rj) > 0:
            counts = np.bincount(np.int_(Rj[np.where(Rj != 0)[0]]))
            Y_tilde[j] = np.argmax(counts)
        else:
            Rj = R[:,j]
            counts = np.bincount(np.int_(Rj[np.where(Rj != 0)[0]]))
            Y_tilde[j] = np.argmax(counts)
    # print(Y, Y_tilde)
    acc = np.count_nonzero(Y_tilde == Y)/m
    return acc

def accuracy_computer_threshold_realdata(R, Y, agent_w):
    R = np.array(R)
    R_w = R[np.where(agent_w == 1)[0]]
    m = np.size(Y)
    Y_tilde = np.zeros(m)
    for j in range(m):
        Rj = R_w[:,j]
        if np.count_nonzero(Rj) > 0:
            counts = np.bincount(np.int_(Rj[np.where(Rj != 0)[0]]))
            if len(counts) < 6:
                counts = np.concatenate((counts,np.zeros(6-len(counts))))
            ct1 = counts[1]+counts[2]
            ct2 = counts[4]+counts[5]
            if ct1 > ct2:
                Y_tilde[j] = 1
            elif ct2 >= ct1:
                Y_tilde[j] = 2
        else:
            Rj = R[:,j]
            counts = np.bincount(np.int_(Rj[np.where(Rj != 0)[0]]))
            if len(counts) < 6:
                counts = np.concatenate((counts,np.zeros(6-len(counts))))
            ct1 = counts[1]+counts[2]
            ct2 = counts[4]+counts[5]
            if ct1 > ct2:
                Y_tilde[j] = 1
            elif ct2 >= ct1:
                Y_tilde[j] = 2
    # print(Y, Y_tilde)
    acc = np.count_nonzero(Y_tilde == Y)/m
    return acc

def accuracy_computer_weighted(R, Y, weight, para):
    R = np.array(R)
    Y_tilde = weighted_majority_vote(R, weight, para)
    acc = np.count_nonzero(Y_tilde == Y)/m
    return acc

def weighted_majority_vote(R, y, para):
    S = para.S
    m = para.m
    vote = np.zeros((m, S))
    for j in range(m):
        for s in range(1, S+1):
            vote[j,s-1] = np.sum(y[np.where(R[:,j] == s)[0]])
    x = np.argmax(vote, axis = 1) + 1
    return x
        

def iterated_weighted_maj(R_w, R_s, Y, f, k_max, para):
    n_w = np.size(R_w, axis = 0)
    n_s = np.size(R_s, axis = 0)
    y = np.zeros(n_w + n_s)
    x = np.zeros(np.size(R_w, axis = 1))
    R = np.vstack((R_w, R_s))
    if f < 4:
        y_mi = mechanism_matrix_fast(R, f, para)
        y = y_mi - np.min(y_mi)
    elif f == 4:
        y_mi = mechanism_determinant_fast(np.vstack((R_w, R_s)), para)
        y = y_mi - np.min(y_mi)
        
    for k in range(k_max):
        x = weighted_majority_vote(R, y, para)
        if f < 4:
            y_mi = mechanism_matrix_gt(R, f, x, para)
            y = y_mi - np.min(y_mi)
        elif f == 4:
            y_mi = mechanism_determinant_gt(R, x, para)
            y = y_mi - np.min(y_mi)
    
    x = weighted_majority_vote(R, y, para)
    acc = np.count_nonzero(x == Y)/para.m
    t_vs = np.max(y[n_w:n_w+n_s])
    agent_w = np.zeros(n_w+n_s)
    agent_w[y > t_vs] = 1
    return acc, agent_w
# mi = 50
# para_w = Crowdsourcing(w = w, Gamma_w = Gamma_w, Gamma_s = Gamma_s, e = 1, mi = mi, n0 = 10)
# n = math.ceil(m*50/(mi - 2))
# R, Y, agent_e = Report_Generator_2(para_w, [0,0.5,0.5], [int(n*para_w.prior_e[0]),n-int(n*para_w.prior_e[0])-1,1])
# print(accuracy_computer_threshold_realdata(R, Y, agent_e))


"""
Varying Budget
"""
from datetime import datetime
print(datetime.now())

w = np.array([0.19587629, 0.24054983, 0.24742268, 0.3161512])
Gamma = np.array([[0.77056673, 0.12157221, 0.08409506, 0.023766],
                 [0.09083969, 0.7351145, 0.12977099, 0.04427481],
                 [0.03326256, 0.06157113, 0.86624204, 0.03892427],
                 [0.06785509, 0.16388729, 0.09890742, 0.6693502]])
m = 100
mi = 60
Gamma_random = np.ones((4,4))/4
T = 5000
alpha = 0.1 # parameter of the cost function
Rho = [0, 2] # parameter of risk averse level
de = 0.015
n0 = 8
n = math.ceil(m*n0/(mi - 2))
n_s = int(n/5) + 1
Efforts = np.arange(0.5, 1, de)
Thresholds_fraction = np.arange(0.1,0.9,0.05)
Budgets = np.arange(2,30,0.2)

Acc_opt = np.zeros((len(Efforts)))
Acc = np.zeros((len(Efforts),5))

flag = np.zeros((len(Thresholds_fraction),5))
Utility_diff_step = np.zeros((len(Efforts),len(Thresholds_fraction),5,len(Rho),len(Budgets)))
Utility_cost_step = np.zeros((len(Efforts),len(Thresholds_fraction),5,len(Rho),len(Budgets)))
Utility_diff_linear = np.zeros((len(Efforts),5,len(Rho),len(Budgets)))
Utility_cost_linear = np.zeros((len(Efforts),5,len(Rho),len(Budgets)))


for h, eff in enumerate(Efforts):
    c_i = np.array([cost_2(eff, alpha), cost_2(eff+de, alpha)])*(mi/60)
    U_step = np.zeros((len(Thresholds_fraction),5,len(Rho),len(Budgets),2))
    U_linear = np.zeros((5,len(Rho),len(Budgets),2))
    print('effort in eq ', eff)
    for j, e in enumerate([eff, eff+de]):
        print(datetime.now())
        para_w = Crowdsourcing(w = w, Gamma_w = Gamma, Gamma_s = Gamma_random, e = e, mi = mi, n0 = n0, signal = [1,2,3,4])
        para_s = Crowdsourcing(w = w, Gamma_w = Gamma, Gamma_s = Gamma_random, prior_e = [1,0,0], mi = mi, n0 = int(n0/5), signal = [1,2,3,4])
        for l in range(T):
            if l%500 == 0:
                print(j,l)
            R, Y, agent_e = Report_Generator_2(para_w, [0,eff,e], [int(n*para_w.prior_e[0]),n-int(n*para_w.prior_e[0])-1,1])
            P = np.zeros((n,4))
            for k in range(4):
                P[:,k] = mechanism_pairing(R, k, para_w)
                
            if j == 0: ### Compute accuracy
                agent_w = np.zeros((n,4))
                for k in range(4):
                    agent_w[:,k][P[:,k].argsort()[-int(n*0.8):][::-1]] = 1
                    Acc[h,k] += accuracy_computer_threshold(R, Y, agent_w[:,k])/T
                agent_w = agent_e.copy()
                agent_w[agent_e==2] = 1
                Acc_opt[h] += accuracy_computer_threshold(R, Y, agent_w)/T
                        
            for i,t in enumerate(Thresholds_fraction):
                n_w = max(int(n*t),1)
                for k in range(4):
                    for g, B in enumerate(Budgets):
                        a1 = B/n_w
                        for r, rho in enumerate(Rho):
                            if np.count_nonzero(agent_e[P[:,k].argsort()[-n_w:][::-1]] == 2) == 1:
                                if a1 >= c_i[j]:
                                    U_step[i,k,r,g,j] += a1/T
                                else:
                                    U_step[i,k,r,g,j] += (a1 - rho*np.square(c_i[j]-a1))/T
                            else:
                                U_step[i,k,r,g,j] += -rho*np.square(c_i[j])/T
            for k in range(4):
                P_linear = (P[:,k]-min(P[:,k]))/(max(P[:,k]-min(P[:,k])))
                P_linear_sum = np.sum(P_linear)
                for g, B in enumerate(Budgets):
                    a1 = B/P_linear_sum
                    P_i = P_linear[np.where(agent_e == 2)[0]]*a1
                    for r, rho in enumerate(Rho):
                        if P_i >= c_i[j]:
                            U_linear[k,r,g,j] += P_i/T
                        else:
                            U_linear[k,r,g,j] += (P_i - rho*np.square(c_i[j]-P_i))/T
                      
    for i in range(len(Thresholds_fraction)):
        for k in range(4):
            for r, rho in enumerate(Rho):
                for g, B in enumerate(Budgets):
                    Utility_diff_step[h,i,k,r,g] = U_step[i,k,r,g,1] - U_step[i,k,r,g,0]
                    Utility_cost_step[h,i,k,r,g] = U_step[i,k,r,g,0] - c_i[0]
#         if i == 0:
#             print('utility, 10%, tvd, B=2', Utility_cost_step[h,i,0,0,0])
#             print('utility, 10%, tvd, B=10', Utility_cost_step[h,i,0,0,40])
#             print('utility, 10%, tvd, B=20', Utility_cost_step[h,i,0,0,180])
    for k in range(4):
        for r, rho in enumerate(Rho):
            for g, B in enumerate(Budgets):
                Utility_diff_linear[h,k,r,g] = U_linear[k,r,g,1] - U_linear[k,r,g,0]
                Utility_cost_linear[h,k,r,g] = U_linear[k,r,g,0] - c_i[0]
#     print('utility, linear, tvd, B=10', Utility_cost_linear[h,0,0,80])



np.save('Results/mi_'+str(mi)+'_n0_'+str(n0)+'_w2/Pairing/Utility_diff_step.npy',Utility_diff_step)
np.save('Results/mi_'+str(mi)+'_n0_'+str(n0)+'_w2/Pairing/Utility_cost_step.npy',Utility_cost_step)
np.save('Results/mi_'+str(mi)+'_n0_'+str(n0)+'_w2/Pairing/Utility_diff_linear.npy',Utility_diff_linear)
np.save('Results/mi_'+str(mi)+'_n0_'+str(n0)+'_w2/Pairing/Utility_cost_linear.npy',Utility_cost_linear)
np.save('Results/mi_'+str(mi)+'_n0_'+str(n0)+'_w2/Pairing/Acc.npy',Acc)
np.save('Results/mi_'+str(mi)+'_n0_'+str(n0)+'_w2/Pairing/Acc_opt.npy',Acc_opt)