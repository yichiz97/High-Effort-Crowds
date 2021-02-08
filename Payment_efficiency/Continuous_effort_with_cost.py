#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:46:07 2021

@author: yichiz
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from random import choices
from random import sample
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM
from seaborn import scatterplot as scatter
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture
import cvxpy as cp
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.optimize import fsolve
from scipy import optimize
from pynverse import inversefunc

#%%
"""
Model setup
"""
n = 100
mi = 30
m = 200
prior_e = [0.2, 0.8]

# World 1: signal space 5, clear gap between shirker and worker
# signal = [1, 2, 3, 4, 5]
# S = len(signal)
# w = np.array([0.2, 0.25, 0.25, 0.15, 0.15])
# Gamma = np.array([[0.8, 0.1, 0.04, 0.04, 0.02], 
#           [0.1, 0.8, 0.08, 0, 0.02],
#           [0.01, 0.08, 0.82, 0.06, 0.03],
#           [0, 0, 0.08, 0.85, 0.07],
#           [0, 0.01, 0.01, 0.08, 0.9]])
# Gamma_shirking = np.array([[0.26, 0.34, 0.26, 0.08, 0.06], 
#                             [0.24, 0.3, 0.3, 0.11, 0.05],
#                             [0.1, 0.28, 0.24, 0.26, 0.12],
#                             [0.06, 0.13, 0.28, 0.3, 0.23],
#                             [0.12, 0.08, 0.22, 0.3, 0.28]])
# Gamma_random = np.ones((S,S))/S

# # World 2: signal space 5, unclear gap between shirker and worker
signal = [1, 2, 3, 4, 5]
S = len(signal)
w = np.array([0.25, 0.2, 0.15, 0.15, 0.25])
Gamma = np.array([[0.62, 0.22, 0.1, 0.04, 0.02], 
          [0.14, 0.63, 0.12, 0.09, 0.02],
          [0.05, 0.16, 0.58, 0.14, 0.07],
          [0.02, 0.06, 0.15, 0.61, 0.16],
          [0.02, 0.05, 0.11, 0.18, 0.64]])
Gamma_shirking = np.array([[0.34, 0.26, 0.26, 0.08, 0.06], 
                            [0.24, 0.33, 0.27, 0.11, 0.05],
                            [0.1, 0.24, 0.28, 0.26, 0.12],
                            [0.06, 0.13, 0.28, 0.3, 0.23],
                            [0.1, 0.07, 0.21, 0.29, 0.33]])
Gamma_random = np.ones((S,S))/S

# World 3: signal space 3,  clear gap between shirker and worker
# signal = [1, 2, 3]
# S = len(signal)
# w = np.array([0.4, 0.3, 0.3])
# Gamma = np.array([[0.85, 0.1, 0.05], 
#           [0.12, 0.8, 0.08],
#           [0.04, 0.08, 0.88]])
# Gamma_shirking = np.array([[0.35, 0.35, 0.3], 
#           [0.32, 0.36, 0.32],
#           [0.28, 0.35, 0.37]])

# World 4: signal space 2, unclear gap between shirker and worker
# signal = [1, 2]
# S = len(signal)
# w = np.array([0.5, 0.5])
# Gamma = np.array([[0.85, 0.15], 
#           [0.12, 0.88]])
# Gamma_shirking = np.array([[0.66, 0.34], 
#           [0.4, 0.6]])

# Gamma_e = []
# for e in E:
#     Gamma_e.append(Gamma*e+Gamma_shirking*(1-e))

#%%
"""
Generate reports
"""

def Report_Generator_Uniform(n, mi, m, prior_e, signal, w, Gamma_e):
    agent = np.arange(len(prior_e))
    agent_e = choices(agent, prior_e, k = n)
    Y = choices(list(range(1, len(w)+1)), w, k = m)
    
    R = []
    for i in agent_e:
        Xi = np.zeros(m)
        Gamma_i = Gamma_e[i]
        task_i = np.array(sample(range(m), mi))
        for j in task_i:
            Xi[j] = choices(signal, Gamma_i[Y[j]-1])[0]
        R.append(Xi)
    
    return (R, Y, np.array(agent_e))

#%%
"""
Compute pairing MI
"""

def Soft_pred_to_MI(Marginal, pred, b, p, q, Y_T):
    # Mi = np.zeros(5)
    # K_b = np.zeros(5)
    # K_p = np.zeros(5)
    Mi = np.zeros(4)
    K_b = np.zeros(4)
    K_p = np.zeros(4)
    predict_b = pred[b]
    predict_p = pred[p]

    if Marginal[Y_T[b]-1] != 0:
        Jp_b = predict_b[Y_T[b]-1]/Marginal[Y_T[b]-1]
        if Jp_b > 1:
            K_b[0] = 0.5
        elif Jp_b < 1:
            K_b[0] = -0.5
        
        if 1 + np.log(Jp_b) < -10:
            K_b[1] = -10
        elif 1 + np.log(Jp_b) > 10:
            K_b[1] = 10
        else:
            K_b[1] = 1 + np.log(Jp_b)
        
        K_b[2] = 2*(Jp_b - 1)
        
        if 1 - 1/np.sqrt(Jp_b) < -10:
            K_b[3] = -10
        else:
            K_b[3] = 1 - 1/np.sqrt(Jp_b)
        
       # K_b[4] = 4*np.power(Jp_b - 1, 3)
        
       # K_b[5] = 6*np.power(Jp_b - 1, 5)
        
    if Marginal[Y_T[q]-1] != 0:
        Jp_p = predict_p[Y_T[q]-1]/Marginal[Y_T[q]-1]
        if Jp_p > 1:
            K_p[0] = 0.5
        elif Jp_p < 1:
            K_p[0] = -0.5
        
        kl = 1 + np.log(Jp_p)
        if kl < -10:
            K_p[1] = -10
        elif kl > 10:
            K_p[1] = 10
        else:
            K_p[1] = kl
        
        K_p[2] = 2*(Jp_p - 1)
        
        heli = 1 - 1/np.sqrt(Jp_p)
        if heli < -10:
            K_p[3] = -10
        else:
            K_p[3] = heli
        
       # K_p[4] = 4*np.power(Jp_p - 1, 3)
        
   #     K_p[5] = 6*np.power(Jp_p - 1, 5)
    
    Mi[0] = K_b[0] - K_p[0]
    Mi[1] = K_b[1] - np.exp(K_p[1]-1)
    Mi[2] = K_b[2] - np.square(K_p[2])/4 - K_p[2]
    Mi[3] = K_b[3] - K_p[3]/(1-K_p[3])
   # Mi[4] = K_b[4] - np.power(np.abs(K_p[4])/4, 1/3)*np.abs(K_p[4])*3/4 - K_p[4]
  #  Mi[5] = K_b[5] - np.power(np.abs(K_p[5])/6, 1/5)*np.abs(K_p[5])*5/6 - K_p[5]
    return Mi


def soft_predictor_learner(X):
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

def LR_learner_2(X, Y):
    Y = np.array(Y)
    T = np.size(np.where(Y != 0)[0])*10
    index_i = np.where(Y != 0)[0]
    Yi = Y[index_i]
    M = np.zeros(4)
    
    Y_int = Y.copy().astype(int)
    pred = soft_predictor_learner(X)
    Marginal = np.zeros(S)
    for j in signal:
        Marginal[j-1] = np.count_nonzero(Yi == j)/len(Yi)
    for j in range(T):
        b,q = sample(list(index_i), 2)
        p = sample(range(len(Y)), 1)[0]
        while p == b or p == q:
            p = sample(range(len(Y)),1)[0]
        Mi = Soft_pred_to_MI(Marginal, pred, b, p, q, Y_int)
        M += Mi
    M = M/(T)

    return M


# Pairing mechanism, can be used as black box. Now it requires ground truth, which is a perfect predictor of agents' reports.
# In reality, Gr can be replaced by a soft learning algorithm.
def mechanism_pairing_learning_3(X):
    n = len(X)

    X = np.array(X)
    U = np.zeros((n,4))
    for i in range(n):
        # if i%20 ==0:
        #     print("agent no. ", i)
        X_ni = np.vstack((X[0:i], X[i+1:n]))
        #U[i] = LR_learner(X_ni, X[i], Gr)
        U[i] = LR_learner_2(X_ni, X[i])

    return U

def LR_learner_3(pred, Y):
    Y = np.array(Y)
    T = np.size(np.where(Y != 0)[0])*10
    index_i = np.where(Y != 0)[0]
    Yi = Y[index_i]
    M = np.zeros(4)
    
    Y_int = Y.copy().astype(int)
    Marginal = np.zeros(S)
    for j in signal:
        Marginal[j-1] = np.count_nonzero(Yi == j)/len(Yi)
    for j in range(T):
        b,q = sample(list(index_i), 2)
        p = sample(range(len(Y)), 1)[0]
        while p == b or p == q:
            p = sample(range(len(Y)),1)[0]
        Mi = Soft_pred_to_MI(Marginal, pred, b, p, q, Y_int)
        M += Mi
    M = M/(T)

    return M

def mechanism_pairing_4(Xa, Xb, Xs):
    ns = np.size(Xs, axis=0)
    na = np.size(Xa, axis=0)
    Us = np.zeros((ns, 4))
    for i in range(ns):
        X_ni = np.vstack((Xs[0:i], Xs[i+1:ns], Xb))
        Us[i] = LR_learner_2(X_ni, Xs[i])
    pred = soft_predictor_learner(np.vstack((Xs, Xb)))
    Ua = np.zeros((na, 4))
    for i in range(na):
        Ua[i] = LR_learner_3(pred, Xa[i])
    return Ua, Us

# R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e])
# R = np.array(R)
# mechanism_pairing_4(R[0:int(n/3)], R[0:2*int(n/3)], R[0:int(n/3)])


#%%
"""
Cost function
"""

def cost(x, a):
    b = 1/(1/(1+np.exp(-a))-1/(1+np.exp(-0.1*a)))
    c = 1 - b/(1+np.exp(-a))
    return np.log((x-c)/(b+c-x))/a

"""
Compute accuracy: majority vote
"""

def accuracy_computer(R, Y):
    R = np.array(R)
    m = np.size(Y)
    Y_tilde = np.zeros(m)
    for j in range(m):
        Rj = R[:,j]
        counts = np.bincount(np.int_(Rj[np.where(Rj != 0)[0]]))
        Y_tilde[j] = np.argmax(counts)
    # print(Y, Y_tilde)
    acc = np.count_nonzero(Y_tilde == Y)/m
    return acc


#%%
"""
Accuracy vs cost
Varying the number of agents
"""

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
#%%
"""
Varying mi
Gap between virtual_shirker step and opt step
"""

n = 50
m = 100
M = [20, 30, 40, 50, 80]
Effort = np.arange(0, 1.01, 0.01)
Thresholds = np.arange(0.2, 1, 0.01)
Amplitudes = np.arange(0.5, 2.01, 0.02)
T = 80
alpha = 5
U_vs = np.zeros((len(M), len(Effort), len(Amplitudes), 4))
Payments_vs = np.zeros((len(M), len(Effort), len(Amplitudes), 4))
U_t = np.zeros((len(M), len(Effort), len(Thresholds), len(Amplitudes), 4))
Payments_t = np.zeros((len(M), len(Effort), len(Thresholds), len(Amplitudes), 4))
Acc_vs = np.zeros((len(M), len(Effort), 4))
Acc_t = np.zeros((len(M), len(Effort), len(Thresholds), 4))

Amplitude_vs = np.zeros((len(M), len(Effort), 4))
Threshold_vs = np.zeros((len(M), len(Effort), 4))

for l in range(T):
    for no_m, mi in enumerate(M):
        print('mi = ',mi, 'round ',l)
        for i, e in enumerate(Effort):
            Gamma_e = e*Gamma + (1-e)*Gamma_shirking
            while True:
                R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e, Gamma])
                if np.count_nonzero(np.count_nonzero(np.array(R), axis = 0)) == m:
                    break
                
            R_s, _, _ = Report_Generator_Uniform(n_s, mi, m, [1], signal, w, [Gamma_random])
            Amplitude_vs[no_m, i] += empirical_gap_pairing(R, R_s)/T
            P_all = mechanism_pairing_learning_3(np.vstack((R, R_s)))
            # P_all = mechanism_matrix_learning(np.vstack((R, R_s)))
            P = P_all[0:n]
            t_vs = np.max(P_all[n:n+n_s], axis = 0)
            Threshold_vs += t_vs/T
            c = cost(e, alpha)
            for k in range(4):
                agent_w = np.zeros(n)
                agent_w[P[:,k]>t_vs[k]] = 1
                Acc_vs[no_m,i,k] += accuracy_computer_threshold(R, Y, agent_w)/T
            for h, a in enumerate(Amplitudes):
                P_step_vs = np.zeros((n,4)) + 0.1
                P_step_vs[P>t_vs] = a
                U_vs[no_m, i, h] += (np.average(P_step_vs[agent_e == 1], axis = 0) - c)/T
                Payments_vs[no_m,i,h] += np.average(P_step_vs, axis = 0)*mi/T
            
            for j, t in enumerate(Thresholds):
                t_prime = (1-t)*np.min(P, axis = 0) + t*np.max(P, axis = 0)
                for k in range(4):
                    agent_w = np.zeros(n)
                    agent_w[P[:,k]>t_prime[k]] = 1
                    Acc_t[no_m,i,j,k] += accuracy_computer_threshold(R, Y, agent_w)/T
                for h, a in enumerate(Amplitudes):
                    P_step_t = np.zeros((n,4)) + 0.1
                    P_step_t[P>t_prime] = a
                    U_t[no_m, i, j, h] += (np.average(P_step_t[agent_e == 1], axis = 0) - c)/T
                    Payments_t[no_m,i,j,h] += np.average(P_step_t, axis = 0)*mi/T
                    
np.save('World2_pairing/U_vs.npy',U_vs)                    
np.save('World2_pairing/Payments_vs.npy',Payments_vs)
np.save('World2_pairing/U_t.npy',U_t)
np.save('World2_pairing/Payments_t.npy',Payments_t)
np.save('World2_pairing/Acc_vs.npy',Acc_vs)
np.save('World2_pairing/Acc_t.npy',Acc_t)
np.save('World2_pairing/Amplitude_vs.npy',Amplitude_vs)
np.save('World2_pairing/Threshold_vs.npy',Threshold_vs)
#%%
# Acc_goal = 0.99
# Payment_min_vs = np.zeros((len(M), 4))
# Effort_min_vs = np.zeros((len(M), 4))
# Amplitude_min_vs_opt = np.zeros((len(M), 4))
# Amplitude_vs_emp = np.zeros((len(M), 4))
# for no_m in range(4):
#     Effort_emp = np.argmax(U_vs[no_m], axis = 0)
#     for k in range(4):
#         if np.count_nonzero(Acc_vs[no_m,:,k] >= Acc_goal) == 0:
#             amplitude_range_vs = []
#         else:
#             index_acc = next(x for x in Acc_vs[no_m,:,k] if x >= Acc_goal)
#             min_effort_vs = np.where(Acc_vs[no_m,:,k] == index_acc)[0][0]
#             amplitude_range_vs = np.where(Effort_emp[:,k] >= min_effort_vs)[0]
#         pay_min_k = 1000
#         for h in amplitude_range_vs:
#             if Payments_vs[no_m,Effort_emp[h,k],h,k] < pay_min_k:
#                 pay_min_k = Payments_vs[no_m,Effort_emp[h,k],h,k]
#                 eff_min_k = Effort[Effort_emp[h,k]]
#                 amp_min_k = Amplitudes[h]
#         Payment_min_vs[no_m, k] = pay_min_k
#         Effort_min_vs[no_m, k] = eff_min_k
#         Amplitude_min_vs_opt[no_m, k] = amp_min_k
#         Amplitude_vs_emp[no_m, k] = Amplitude_vs[no_m, int(eff_min_k*100), k]

# Payment_min_opt = np.zeros((len(M), 4))
# Effort_min_opt = np.zeros((len(M), 4))
# Amplitude_min_opt = np.zeros((len(M), 4))
# Threshold_opt = np.zeros((len(M), 4))
# for no_m in range(4):
#     Effort_opt = np.argmax(U_t[no_m], axis = 0)
#     for k in range(4):
#         pay_min_k = 1000
#         for j,t in enumerate(Thresholds):
#             if np.count_nonzero(Acc_t[no_m,:,j,k] >= Acc_goal) == 0:
#                 amplitude_range_t = []
#             else:
#                 index_acc = next(x for x in Acc_t[no_m,:,j,k] if x >= Acc_goal)
#                 min_effort_t = np.where(Acc_t[no_m,:,j,k] == index_acc)[0][0]
#                 amplitude_range_t = np.where(Effort_opt[j,:,k] >= min_effort_t)[0]
          
#             for h in amplitude_range_t:
#                 if Payments_t[no_m,Effort_opt[j,h,k],j,h,k] < pay_min_k:
#                     pay_min_k = Payments_t[no_m,Effort_opt[j,h,k],j,h,k]
#                     eff_min_k = Effort[Effort_opt[j,h,k]]
#                     amp_min_k = Amplitudes[h]
#                     ths_min_k = Thresholds[j]
#         Payment_min_opt[no_m, k] = pay_min_k
#         Effort_min_opt[no_m, k] = eff_min_k
#         Amplitude_min_opt[no_m, k] = amp_min_k
#         Threshold_opt[no_m, k] = ths_min_k

# print('Virtual shirker min_payment: ', '\n', Payment_min_vs)
# print('Optimal min_payment: ', '\n', Payment_min_opt)
# print('Optimal Thresholds: ', '\n', Threshold_opt)
# print('Optimal Amplitudes: ', '\n', Amplitude_min_opt)
# print('Optimal elicited efforts: ', '\n', Effort_min_opt)
