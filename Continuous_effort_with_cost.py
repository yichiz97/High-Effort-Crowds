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
signal = [1, 2, 3, 4, 5]
S = len(signal)
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
Gamma_random = np.ones((S,S))/S

# # World 2: signal space 5, unclear gap between shirker and worker
# signal = [1, 2, 3, 4, 5]
# S = len(signal)
# w = np.array([0.1, 0.1, 0.4, 0.2, 0.2])
# Gamma = np.array([[0.62, 0.22, 0.1, 0.04, 0.02], 
#           [0.14, 0.63, 0.12, 0.09, 0.02],
#           [0.05, 0.16, 0.58, 0.14, 0.07],
#           [0.02, 0.06, 0.15, 0.61, 0.16],
#           [0.02, 0.05, 0.11, 0.18, 0.64]])
# Gamma_shirking = np.array([[0.34, 0.26, 0.26, 0.08, 0.06], 
#                             [0.24, 0.33, 0.27, 0.11, 0.05],
#                             [0.1, 0.24, 0.28, 0.26, 0.12],
#                             [0.06, 0.13, 0.28, 0.3, 0.23],
#                             [0.1, 0.07, 0.21, 0.29, 0.33]])

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

Gamma_e = []
for e in E:
    Gamma_e.append(Gamma*e+Gamma_shirking*(1-e))

#%%
"""
Real-data: word wide
"""
import csv

R_real = []
agent_e = []
with open('/Users/yichiz/Personal and Private/Research Projects/Scoring multi-task mechanisms/Data/RealData/word_wide.csv', 'r') as word_wide_data:
    word_wide_data_reader = csv.reader(word_wide_data)
    for row in word_wide_data_reader:
        names = row
        break
    next(word_wide_data_reader)
    for row in word_wide_data_reader:
        if row[1][0] == 'N':
            agent_e.append(0)
        elif row[1][0] == 'S':
            agent_e.append(1)
        elif row[1][0] == 'E':
            agent_e.append(2)
        row = [int(string) for string in row[2:]]
        row[0] = int(row[0])
        
        R_real.append(row)

name = []
for i in names[2:]:
    name.append(i[16:])
name = np.array(name)

Y = np.zeros(len(name)) - 1
with open('/Users/yichiz/Personal and Private/Research Projects/Scoring multi-task mechanisms/Data/RealData/questions.csv', 'r') as ground_truth:
    ground_truth_reader = csv.reader(ground_truth)
    next(ground_truth_reader)
    for row in ground_truth_reader:
        if row[2] == 'appropriate':
            Y[np.where(name == row[0])] = 1
        elif row[2] == 'inappropriate':
            Y[np.where(name == row[0])] = 0
            
R_real = np.array(R_real)
agent_e_real = np.array(agent_e)
R_realdata = R_real.T[Y != -1].T
Y_realdata = Y[Y != -1]

Gamma_real = np.zeros((3,2,5))

for e in range(3):
    Re = R_realdata[agent_e_real == e].T
    for i in range(5):
        Gamma_real[e][0][i] = np.count_nonzero(Re[Y_realdata == 0] == i+1)/np.size(Re[Y_realdata == 0])
        Gamma_real[e][1][i] = np.count_nonzero(Re[Y_realdata == 1] == i+1)/np.size(Re[Y_realdata == 1])

w_real = np.zeros(2)
w_real[0] = np.count_nonzero(Y == 0)/np.size(Y[Y != -1])
w_real[1] = np.count_nonzero(Y == 1)/np.size(Y[Y != -1])

prior_e_real = np.zeros(3)
for i in range(3):
    prior_e_real[i] = np.count_nonzero(agent_e_real == i)/np.size(agent_e_real)

print('Prior of ground truth: ', w_real, '\n')
print('Confusion matrix of non, semi, exp: ', '\n', Gamma_real, '\n')
print('Prior of agents types: ', prior_e_real)
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
Compute matrix MI
"""

def distribution_learner(report, predictor): #another version of distribution_estimator, which takes a report vector and a soft predictor as inputs
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

def MI_computer(P, Q1, Q2): #For matrix mechanism, learm different types of mutual information
    Q = Q2*Q1.reshape(-1, 1)

    MI_tvd = np.sum(np.absolute(P - Q))

    t = P*np.log(P/Q)
    nan_ind = np.isnan(t)
    t[nan_ind] = 0 
    MI_KL = np.sum(t)

    t = P*(np.square((Q/P)-1))
    nan_ind = np.isnan(t)
    t[nan_ind] = 0 
    MI_sqr = np.sum(t)

    t = P*(np.square(np.sqrt(Q/P)-1))
    nan_ind = np.isnan(t)
    t[nan_ind] = 0 
    MI_Hellinger = np.sum(t)
    
    t = P*(np.power((Q/P)-1, 4))
    nan_ind = np.isnan(t)
    t[nan_ind] = 0 
    MI_4 = np.sum(t)
    
    # t = P*(np.power((Q/P)-1, 6))
    # nan_ind = np.isnan(t)
    # t[nan_ind] = 0 
    # MI_6 = np.sum(t)
    
    return np.array([MI_tvd, MI_KL, MI_sqr, MI_Hellinger, MI_4])

def mechanism_matrix_learning(X):
    X = np.array(X)
    n = np.size(X, axis = 0)
    U = np.zeros((n,5))
    for i in range(n):
        X_ni = np.vstack((X[0:i], X[i+1:n]))
        Pi = soft_predictor_learner(X_ni)
        Q_p = np.sum(Pi, axis = 0)/m
        P, Q_a = distribution_learner(X[i], Pi)
        U[i] = MI_computer(P, Q_a, Q_p)
    
    # X1 = X[0:int(n/2)]
    # X2 = X[int(n/2):n]

    # P1 = soft_predictor_learner(X1)
    # P2 = soft_predictor_learner(X2)
    # Q_p1 = np.sum(P1, axis = 0)/m
    # Q_p2 = np.sum(P2, axis = 0)/m
    
    # U1 = np.zeros((int(n/2),5))
    # U2 = np.zeros((n-int(n/2),5))
    # for i in range(np.size(X1, axis = 0)):
    #     P, Q_a = distribution_learner(X1[i], P2)
    #     U1[i] = MI_computer(P, Q_a, Q_p2)
    # for i in range(np.size(X2, axis = 0)):
    #     P, Q_a = distribution_learner(X2[i], P1)
    #     U2[i] = MI_computer(P, Q_a, Q_p1)
    # U = np.vstack((U1, U2))

    return U
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

e = 0.9
Gamma_e = e*Gamma + (1-e)*Gamma_shirking
R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e])
print(accuracy_computer(R, Y))

#%%
"""
Fix amplitude and vary threshold
"""

n = 100
m = 200
mi = 30
T = 50
a = 2
n_s = int(n/5)
Thresholds = np.arange(0, 0.9, 0.02)
Effort = np.arange(0.02, 1.01, 0.02)
#t_vs = np.zeros((len(Effort), 5))
Threshold_effort = np.zeros((len(Thresholds), len(Effort), 5))
for l in range(T):
    print(l)
    for i, e in enumerate(Effort):
        print(e)
        Gamma_e = e*Gamma + (1-e)*Gamma_shirking
        R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e])
        R_s, _, _ = Report_Generator_Uniform(n_s, mi, m, [1], signal, w, [Gamma_shirking])
        P_all = mechanism_pairing_learning_3(np.vstack((R, R_s)))
        P = P_all[0:n]
        #t_vs[i] += np.max(P_all[n:n+n_s], axis = 0)/T
        t_vs = np.max(P_all[n:n+n_s], axis = 0)
        c = cost(e, a)
        
        for j, t in enumerate(Thresholds):
            #t_prime = (1-t)*np.min(P, axis = 0) + t*np.max(P, axis = 0)
            t_prime = (1-t)*t_vs + t*np.max(P, axis = 0)
            P_step = np.zeros((n,5))
            P_step[P>t_prime] = 2
            U_t = np.average(P_step[agent_e == 1], axis = 0) - c
            Threshold_effort[j, i] += U_t/T

#%%
MI = ['TVD', 'KL', 'Square', 'Hellinger', '4th']
colors = ['orangered', 'navy', 'limegreen', 'goldenrod', 'mediumpurple']
Effort_opt = 0.02 + np.argmax(Threshold_effort, axis = 1)*0.02

plt.figure()
for k in range(5):
    plt.plot(Thresholds, Effort_opt[:,k], color = colors[k], label=MI[k])
plt.xlabel('Threshold')
plt.ylabel('Optimal effort')
plt.title('threshold and effort in NE: Pairing, a = 2, payment = 2')
plt.legend()
plt.show()
#%%
plt. figure()
plt.scatter(P[:,1][agent_e == 0], np.random.uniform(0,1,np.size(np.where(agent_e == 0)[0])), color = 'r')
plt.scatter(P[:,1][agent_e == 1], np.random.uniform(0,1,np.size(np.where(agent_e == 1)[0])), color = 'b')

#%%
"""
Fix threshold and vary amplitude
"""

n = 100
m = 200
mi = 200
T = 50
alpha = 2
n_s = int(n/5)
Amplitude = np.arange(0.5, 5, 0.1)
Effort = np.arange(0.02, 1.01, 0.02)
#t_vs = np.zeros((len(Effort), 5))
Amplitude_effort = np.zeros((len(Amplitude), len(Effort), 5))
Payment_overall = np.zeros((len(Amplitude), len(Effort), 5))
Accuracy = np.zeros(len(Effort))
for l in range(T):
    print(l)
    for i, e in enumerate(Effort):
        Gamma_e = e*Gamma + (1-e)*Gamma_shirking
        while True:
            R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e])
            if np.count_nonzero(np.count_nonzero(np.array(R), axis = 0)) == m:
                break
                
        R_s, _, _ = Report_Generator_Uniform(n_s, mi, m, [1], signal, w, [Gamma_shirking])
        # P_all = mechanism_pairing_learning_3(np.vstack((R, R_s)))
        P_all = mechanism_matrix_learning(np.vstack((R, R_s)))
        P = P_all[0:n]
        #t_vs[i] += np.max(P_all[n:n+n_s], axis = 0)/T
        t_vs = np.max(P_all[n:n+n_s], axis = 0)
        c = cost(e, alpha)
        Accuracy[i] += accuracy_computer(R, Y)/T
        
        for j, a in enumerate(Amplitude):
            P_step = np.zeros((n,5)) + 0.1
            P_step[P>t_vs] = a
            U_t = np.average(P_step[agent_e == 1], axis = 0) - c
            Amplitude_effort[j, i] += U_t/T
            Payment_overall[j, i] += np.sum(P_step, axis = 0)/T



#%%
MI = ['TVD', 'KL', 'Square', 'Hellinger', '4th']
colors = ['orangered', 'navy', 'limegreen', 'goldenrod', 'mediumpurple']
Effort_opt = 0.02 + np.argmax(Amplitude_effort, axis = 1)*0.02

plt.figure()
for k in range(5):
    plt.plot(Amplitude, Effort_opt[:,k], color = colors[k], label=MI[k])
plt.xlabel('Amplitude')
plt.ylabel('Optimal effort')
plt.title('Amplitude and opt effort: Pairing, alpha = 1')
plt.legend()
plt.show()

Payment_opt = np.zeros((len(Amplitude), 5))
Acc_opt = np.zeros((len(Amplitude), 5))
for j, a in enumerate(Amplitude):
    for k in range(5):
        Payment_opt[j,k] = Payment_overall[j][np.argmax(Amplitude_effort[j,:,k])][k]
        Acc_opt[j,k] = Accuracy[np.argmax(Amplitude_effort[j,:,k])]

plt.figure()
for k in range(5):
    plt.plot(Amplitude, Payment_opt[:,k], color = colors[k], label=MI[k])
plt.xlabel('Amplitude')
plt.ylabel('Total payments')
plt.title('Total_payments_opt_effort vs Amplitude: Pairing, alpha = 1')
plt.legend()
plt.show()

plt.figure()
for k in range(5):
    plt.plot(Amplitude, Acc_opt[:,k], color = colors[k], label=MI[k])
plt.xlabel('Amplitude')
plt.ylabel('Accuracy')
plt.title('Accuracy_opt_effort vs Amplitude: Pairing, alpha = 1')
plt.legend()
plt.show()


#%%
np.savetxt("OptimalEffort_S3_n100_m200_mi30_ns20_alpha1.csv", Effort_opt, delimiter=",")
np.savetxt("TotalPayments_S3_n100_m200_mi30_ns20_alpha1.csv", Payment_opt, delimiter=",")
np.savetxt("Accuracy_S3_n100_m200_mi30_ns20_alpha1.csv", Acc_opt, delimiter=",")

#%%
"""
Empirical step function
Compare the optimal effort elicited by the empirical step function and the optimal step function under the same total payments
"""
def empirical_gap_pairing(R, R_s):
    R = np.array(R)
    R_s = np.array(R_s)
    n = np.size(R, axis = 0)
    n_s = np.size(R_s, axis = 0)
    n_3 = int(n/3)
    n_s_2 = int(n_s/2)
    R_A = R[0:n_3]
    R_B = R[n_3:n_3*2]
    R_C = R[n_3*2:n]
    R_s_B = R_s[0:n_s_2]
    R_s_C = R_s[n_s_2:n_s]
    
    # U_AB = np.zeros((n_3, 5))
    # U_sB = np.zeros((n_s_2, 5))
    # for i, Ri in enumerate(R_A):
    #     #print(i, ' / ', np.size(R_A,axis = 0))
    #     U_mi = mechanism_pairing_learning_3(np.vstack((Ri, R_B, R_s_B)))
    #     U_AB[i] = U_mi[0]
    #     U_sB += U_mi[n_3+1:n_3+1+n_s_2]/n_3
    U_AB, U_sB = mechanism_pairing_4(R_A, R_B, R_s_B)
    t_B = np.max(U_sB, axis = 0)
    e_AB = np.zeros((n_3, 4))
    e_AB[U_AB > t_B] = 1
    
    
    # U_AC = np.zeros((n_3, 5))
    # U_sC = np.zeros((n_s-n_s_2, 5))
    # for i, Ri in enumerate(R_A):
    #     U_mi = mechanism_pairing_learning_3(np.vstack((Ri, R_C, R_s_C)))
    #     U_AC[i] = U_mi[0]
    #     U_sC += U_mi[n-2*n_3+1:n-2*n_3+1+n_s-n_s_2]/(n_3)
    U_AC, U_sC = mechanism_pairing_4(R_A, R_C, R_s_C)
    t_C = np.max(U_sC, axis = 0)
    e_AC = np.zeros((n_3, 4))
    e_AC[U_AC > t_C] = 1
    
    a = np.zeros(4)
    for j in range(4):
        n_s_hat = np.size(np.intersect1d(np.where(e_AB[:,j] == 0), np.where(e_AC[:,j] == 0)))
        n_s_B = np.size(np.where(e_AB[:,j] == 0))
        n_s_C = np.size(np.where(e_AC[:,j] == 0))
        p_i = (n_s_B+n_s_C-2*n_s_hat)/(2*n_3)
        a[j] = 1/(1-p_i)
    return a
#%%
n = 100
m = 200
mi = 30
T = 50
alpha = 5
n_s = int(n/4)
Effort = np.arange(0, 1.01, 0.02)
Thresholds = np.arange(0.01, 1, 0.02)
U_effort = np.zeros((len(Effort), 5))
U_threshold_effort = np.zeros((len(Effort), len(Thresholds), 5))
Payment_overall = np.zeros((len(Effort), 5))
Accuracy = np.zeros(len(Effort))
for l in range(T):
    print(l)
    for i, e in enumerate(Effort):
        Gamma_e = e*Gamma + (1-e)*Gamma_shirking
        while True:
            R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e])
            if np.count_nonzero(np.count_nonzero(np.array(R), axis = 0)) == m:
                break
                
        R_s, _, _ = Report_Generator_Uniform(n_s, mi, m, [1], signal, w, [Gamma_shirking])
        a_vs = empirical_gap(R, R_s)
        P_all = mechanism_pairing_learning_3(np.vstack((R, R_s)))
        P = P_all[0:n]
        t_vs = np.max(P_all[n:n+n_s], axis = 0)
        c = cost(e, alpha)
        Accuracy[i] += accuracy_computer(R, Y)/T
        P_step_emp = np.zeros((n,5)) + 0.1
        for k in range(5):
            P_step_emp[:,k][P[:,k]>t_vs[k]] = a_vs[k]
        U_vs = np.average(P_step_emp[agent_e == 1], axis = 0) - c
        U_effort[i] += U_vs/T
        P_overall = np.sum(P_step_emp, axis = 0)
        Payment_overall[i] += P_overall/T
        
        for j, t in enumerate(Thresholds):
            t = (1-t)*t_vs + t*np.max(P, axis = 0)
            P_step = np.zeros((n,5)) + 0.1
            for k in range(5):
                P_step[:,k][P[:,k]>t[k]] = (P_overall[k] - 0.1*(n - np.count_nonzero(P[:,k]>t[k], axis = 0)))/np.count_nonzero(P[:,k]>t[k], axis = 0)
            U_t = np.average(P_step[agent_e == 1], axis = 0) - c
            U_threshold_effort[i, j] += U_t/T

#%%
save_text = np.zeros((5,5))
save_text[0] = np.argmax(U_effort, axis = 0)*0.02
save_text[1] = np.max(np.argmax(U_threshold_effort, axis = 0)*0.02, axis = 0)
for k in range(5):
    save_text[2,k] = Payment_overall[np.argmax(U_effort[:,k]), k]
    save_text[3,k] = Accuracy[np.argmax(U_effort[:,k])]
    save_text[4,k] = Accuracy[np.max(np.argmax(U_threshold_effort[:,:,k], axis = 0))]
print(save_text)
np.savetxt("Results_World2_n100_m200_mi30_ns20_alpha5.csv", save_text, delimiter=",")
#%%
"""
Plot effort_accuracy curve
"""
plt.plot(Effort, Accuracy, color = 'royalblue')
plt.title('Majority vote accuary sv effort')
plt.xlabel('Effort level')
plt.ylabel('Accuracy')

#%%
"""
Opt payment efficiency
"""
n = 100
m = 200
mi = 30
T = 80
alpha = 5
n_s = int(n/4)
Effort = np.arange(0, 1.01, 0.01)
Amplitude = np.arange(0.5, 1.82, 0.02)
U_effort = np.zeros((len(Amplitude), len(Effort), 5))
U_effort_emp = np.zeros((len(Effort), 5))
Payment_average = np.zeros((len(Amplitude), len(Effort), 5))
Accuracy = np.zeros(len(Effort))
Amplitude_vs = np.zeros((len(Effort), 5))
for l in range(T):
    print(l)
    for i, e in enumerate(Effort):
        Gamma_e = e*Gamma + (1-e)*Gamma_shirking
        while True:
            R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e])
            if np.count_nonzero(np.count_nonzero(np.array(R), axis = 0)) == m:
                break
                
        R_s, _, _ = Report_Generator_Uniform(n_s, mi, m, [1], signal, w, [Gamma_shirking])
        a_vs = empirical_gap_pairing(R, R_s)
        Amplitude_vs[i] += a_vs/T
        P_all = mechanism_pairing_learning_3(np.vstack((R, R_s)))
        # P_all = mechanism_matrix_learning(np.vstack((R, R_s)))
        P = P_all[0:n]
        t_vs = np.max(P_all[n:n+n_s], axis = 0)
        c = cost(e, alpha)
        Accuracy[i] += accuracy_computer(R, Y)/T
        P_step_emp = np.zeros((n,5)) + 0.1
        for k in range(5):
            P_step_emp[:,k][P[:,k]>t_vs[k]] = a_vs[k]
        U_vs = np.average(P_step_emp[agent_e == 1], axis = 0) - c
        U_effort_emp[i] += U_vs/T
        
        for j, a in enumerate(Amplitude):
            P_step = np.zeros((n,5)) + 0.1
            P_step[P>t_vs] = a
            U_t = np.average(P_step[agent_e == 1], axis = 0) - c
            U_effort[j, i] += U_t/T
            Payment_average[j, i] += np.sum(P_step, axis = 0)/(T*n)
#%%
"""
World 1
"""
Opt_efforts = np.argmax(U_effort, axis = 1)*0.01
Efficiency = np.zeros((len(Amplitude), 5))
for i in range(len(Amplitude)):
    for k in range(5):
        Efficiency[i,k] = Opt_efforts[i,k]/Payment_average[i, int(Opt_efforts[i,k]*100), k]
Opt_efficiency = np.max(Efficiency, axis = 0)
Opt_amplitude = 0.8 + np.argmax(Efficiency, axis = 0)*0.02

Opt_effort = np.zeros(5)
Emp_effort = np.zeros(5)
for k in range(5):
    Opt_effort[k] = Opt_efforts[int((Opt_amplitude[k]-0.8)*50), k]

print('Opt_effort: ', Opt_effort)
print('Opt_efficiency: ', Opt_efficiency)
print('Opt_amplitude: ', Opt_amplitude, '\n')

Amplitude_opt_effort = np.argmax(np.max(U_effort, axis = 1), axis = 0)*0.02 + 0.5
Effort_opt_effort = np.zeros(5)
Efficiency_opt_effort = np.zeros(5)
for k in range(5):
    Effort_opt_effort[k] = Opt_efforts[int((Amplitude_opt_effort[k]-0.5)/0.02), k]
    Efficiency_opt_effort[k] = Efficiency[int((Amplitude_opt_effort[k]-0.5)/0.02), k]
    
print('Effort_opt_effort', Effort_opt_effort)
print('Efficiency_opt_effort', Efficiency_opt_effort)
print('Amplitude_opt_effort', Amplitude_opt_effort, '\n')

Opt_effort_emp = np.argmax(U_effort_emp, axis = 0)*0.01
Opt_amplitude_emp = np.zeros(5)
Opt_efficiency_emp = np.zeros(5)
for k in range(5):
    Opt_amplitude_emp[k] = Amplitude_vs[int(Opt_effort_emp[k]*100), k]
    Opt_efficiency_emp[k] = Efficiency[int(round((Opt_amplitude_emp[k]-0.5)/0.02)), k]

print('Emp_effort: ', Opt_effort_emp)
print('Emp_efficiency: ', Opt_efficiency_emp)
print('Emp_amplitude: ', Opt_amplitude_emp)
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


n = 30
m = 200
mi = 30
T = 80
alpha = 5
n_s = int(n/4)
Cost = np.arange(0.1, 1.01, 0.01)
cost_a = (lambda x: cost(x, alpha))
Effort = inversefunc(cost_a, y_values=Cost, domain = [0,1])
Acc = np.zeros((len(Effort), 5))

for l in range(T):
    print('time', l)
    for i, e in enumerate(Effort):
        Gamma_e = e*Gamma + (1-e)*Gamma_shirking
        while True:
            R, Y, agent_e = Report_Generator_Uniform(n, mi, m, prior_e, signal, w, [Gamma_shirking, Gamma_e])
            if np.count_nonzero(np.count_nonzero(np.array(R), axis = 0)) == m:
                break
                
        R_s, _, _ = Report_Generator_Uniform(n_s, mi, m, [1], signal, w, [Gamma_shirking])
        P_all = mechanism_pairing_learning_3(np.vstack((R, R_s)))
        # P_all = mechanism_matrix_learning(np.vstack((R, R_s)))
        P = P_all[0:n]
        t_vs = np.max(P_all[n:n+n_s], axis = 0)
        for k in range(5):
            agent_w = np.zeros(n)
            agent_w[P[:,k]>t_vs[k]] = 1
            Acc[i,k] += accuracy_computer_threshold(R, Y, agent_w)/T
 

#%%
plt.figure()
MI = ['TVD', 'KL', 'Square', 'Hellinger', '4th']
colors = ['orangered', 'navy', 'limegreen', 'goldenrod', 'mediumpurple']
for k in range(5):
    plt.plot(Cost, Acc[:,k], color = colors[k], label = MI[k])
plt.legend()
plt.title('Pairing, Worlds_1, n_100, alpha_5')
plt.xlabel('Cost')
plt.ylabel('Accuracy')
#%%
"""
Varying n
Gap between virtual_shirker step and opt step
"""

n = 50
m = 100
M = [20, 30, 40, 50, 80]
Effort = np.arange(0, 1.01, 0.01)
Thresholds = np.arange(0.2, 1, 0.01)
Amplitudes = np.arange(0.5, 2.01, 0.02)
T = 70
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
                    
                    

#%%
Acc_goal = 0.99
Payment_min_vs = np.zeros((len(M), 4))
Effort_min_vs = np.zeros((len(M), 4))
Amplitude_min_vs_opt = np.zeros((len(M), 4))
Amplitude_vs_emp = np.zeros((len(M), 4))
for no_m in range(4):
    Effort_emp = np.argmax(U_vs[no_m], axis = 0)
    for k in range(4):
        if np.count_nonzero(Acc_vs[no_m,:,k] >= Acc_goal) == 0:
            amplitude_range_vs = []
        else:
            index_acc = next(x for x in Acc_vs[no_m,:,k] if x >= Acc_goal)
            min_effort_vs = np.where(Acc_vs[no_m,:,k] == index_acc)[0][0]
            amplitude_range_vs = np.where(Effort_emp[:,k] >= min_effort_vs)[0]
        pay_min_k = 1000
        for h in amplitude_range_vs:
            if Payments_vs[no_m,Effort_emp[h,k],h,k] < pay_min_k:
                pay_min_k = Payments_vs[no_m,Effort_emp[h,k],h,k]
                eff_min_k = Effort[Effort_emp[h,k]]
                amp_min_k = Amplitudes[h]
        Payment_min_vs[no_m, k] = pay_min_k
        Effort_min_vs[no_m, k] = eff_min_k
        Amplitude_min_vs_opt[no_m, k] = amp_min_k
        Amplitude_vs_emp[no_m, k] = Amplitude_vs[no_m, int(eff_min_k*100), k]

Payment_min_opt = np.zeros((len(M), 4))
Effort_min_opt = np.zeros((len(M), 4))
Amplitude_min_opt = np.zeros((len(M), 4))
Threshold_opt = np.zeros((len(M), 4))
for no_m in range(4):
    Effort_opt = np.argmax(U_t[no_m], axis = 0)
    for k in range(4):
        pay_min_k = 1000
        for j,t in enumerate(Thresholds):
            if np.count_nonzero(Acc_t[no_m,:,j,k] >= Acc_goal) == 0:
                amplitude_range_t = []
            else:
                index_acc = next(x for x in Acc_t[no_m,:,j,k] if x >= Acc_goal)
                min_effort_t = np.where(Acc_t[no_m,:,j,k] == index_acc)[0][0]
                amplitude_range_t = np.where(Effort_opt[j,:,k] >= min_effort_t)[0]
          
            for h in amplitude_range_t:
                if Payments_t[no_m,Effort_opt[j,h,k],j,h,k] < pay_min_k:
                    pay_min_k = Payments_t[no_m,Effort_opt[j,h,k],j,h,k]
                    eff_min_k = Effort[Effort_opt[j,h,k]]
                    amp_min_k = Amplitudes[h]
                    ths_min_k = Thresholds[j]
        Payment_min_opt[no_m, k] = pay_min_k
        Effort_min_opt[no_m, k] = eff_min_k
        Amplitude_min_opt[no_m, k] = amp_min_k
        Threshold_opt[no_m, k] = ths_min_k

print('Virtual shirker min_payment: ', '\n', Payment_min_vs)
print('Optimal min_payment: ', '\n', Payment_min_opt)
print('Optimal Thresholds: ', '\n', Threshold_opt)
print('Optimal Amplitudes: ', '\n', Amplitude_min_opt)
print('Optimal elicited efforts: ', '\n', Effort_min_opt)

#%%
no_m = 0
j = 44
k = 2
print(Acc_t[no_m,:,j,k])
print(np.argmax(U_t[no_m], axis = 0)[j,:,k])




































