# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:06:40 2021

@author: yichizhang
"""

from Model_And_Mechanisms import mechanism_matrix_fast, mechanism_determinant_fast, mechanism_pairing, Report_Generator, Crowdsourcing
import numpy as np
import math

w = np.array([0.1, 0.1, 0.4, 0.2, 0.2])
Gamma = np.array([[0.62, 0.22, 0.1, 0.04, 0.02], 
          [0.28, 0.63, 0.09, 0.01, 0.01],
          [0.05, 0.16, 0.58, 0.14, 0.07],
          [0.02, 0.06, 0.15, 0.61, 0.16],
          [0.01, 0.06, 0.02, 0.27, 0.64]])
Gamma_shirking = np.array([[0.34, 0.26, 0.26, 0.08, 0.06], 
                            [0.24, 0.33, 0.27, 0.11, 0.05],
                            [0.1, 0.24, 0.28, 0.26, 0.12],
                            [0.06, 0.13, 0.28, 0.3, 0.23],
                            [0.1, 0.07, 0.21, 0.29, 0.33]])

#%%
"""
Model functions
"""
def cost(x, a):
    b = 1/(1/(1+np.exp(-a))-1/(1+np.exp(-0.1*a)))
    c = 1 - b/(1+np.exp(-a))
    return np.log((x-c)/(b+c-x))/a

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
Experiment: comparing optimal step, virtual shirker step, and iterated virtual shirker step
For matrix and determinant
"""
m = 100
mi = 30
w = np.array([0.2, 0.25, 0.25, 0.15, 0.15])
Gamma_w = np.array([[0.8, 0.1, 0.04, 0.04, 0.02], 
          [0.1, 0.8, 0.08, 0, 0.02],
          [0.01, 0.08, 0.82, 0.06, 0.03],
          [0, 0, 0.08, 0.85, 0.07],
          [0, 0.01, 0.01, 0.08, 0.9]])
Gamma_s = np.array([[0.26, 0.34, 0.26, 0.08, 0.06], 
                            [0.24, 0.3, 0.3, 0.11, 0.05],
                            [0.1, 0.28, 0.24, 0.26, 0.12],
                            [0.06, 0.13, 0.28, 0.3, 0.23],
                            [0.12, 0.08, 0.22, 0.3, 0.28]])
Gamma_random = np.ones((5,5))/5

Effort = np.arange(0, 1.01, 0.02)
Thresholds = np.arange(0.01,1.01,0.02)
Amplitudes = np.arange(0.5, 3.01, 0.02)
N0 = np.arange(5, 35, 5)
T = 80
alpha = 5
U_vs = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
U_vs2 = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
U_t = np.zeros((len(N0), len(Effort), len(Thresholds), len(Amplitudes), 5))
Payments_vs = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
Payments_vs2 = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
Payments_t = np.zeros((len(N0), len(Effort), len(Thresholds), len(Amplitudes), 5))
Acc_vs = np.zeros((len(N0), len(Effort), 5))
Acc_vs2 = np.zeros((len(N0), len(Effort), 5))
Acc_t = np.zeros((len(N0), len(Effort), len(Thresholds), 5))

for no_n0, n0 in enumerate(N0):
    n = math.ceil(m*n0/(mi - 1))
    for l in range(T):
        if l%10 == 0:
            print('n0 ',n0,' ; round ',l)
        for i, e in enumerate(Effort):
            para_w = Crowdsourcing(w = w, Gamma_w = Gamma_w, Gamma_s = Gamma_s, e = e, n0 = n0)
            para_s = Crowdsourcing(w = w, Gamma_w = Gamma_w, Gamma_s = Gamma_random, prior_e = [1,0,0], n0 = int(n0/5))
            R, Y, agent_e = Report_Generator(para_w)
            agent_w = np.ones((n,5))
            agent_w_once = np.ones((n,5))
            P_once = np.zeros((n,5))
            count = 0
            while True:
                agent_w_new = agent_w.copy()
                for k in range(5):
                    
                    n_w = np.count_nonzero(agent_w[:,k])
                    n_s = int(n_w/5) + 1
                    R_s, _, _ = Report_Generator(para_s)
                    R_w = R[np.where(agent_w[:,k] == 1)[0]]
                    if k < 4:
                        P_all = mechanism_matrix_fast(np.vstack((R_w, R_s)), k, para_w)
                    else:
                        P_all = mechanism_determinant_fast(np.vstack((R_w, R_s)), para_w)
                    P = P_all[0:n_w]
                    if n_w == n:
                        P_once[:,k] = P
                    t_vs = np.max(P_all[n_w:n_w+n_s])
                    agent_w_new[np.where(agent_w[:,k] == 1)[0][np.where(P <= t_vs)[0]], k] = 0
                
                if n_w == n:
                    agent_w_once = agent_w_new.copy()
                # print(np.count_nonzero(agent_w, axis = 0))
                if np.array_equal(agent_w_new, agent_w) or count == 100:
                    break
                else:
                    count += 1
                    agent_w = agent_w_new.copy()
            
            
            c = cost(e, alpha)
            for k in range(5):
                Acc_vs[no_n0,i,k] += accuracy_computer_threshold(R, Y, agent_w_once[:,k])/T
                Acc_vs2[no_n0,i,k] += accuracy_computer_threshold(R, Y, agent_w[:,k])/T
            for h, a in enumerate(Amplitudes):
                P_step_vs = np.zeros((n,5)) + 0.1
                P_step_vs[agent_w_once == 1] = a
                U_vs[no_n0,i, h] += (np.average(P_step_vs[agent_e == 1], axis = 0) - c)/T
                Payments_vs[no_n0,i,h] += np.average(P_step_vs, axis = 0)/T
                P_step_vs = np.zeros((n,5)) + 0.1
                P_step_vs[agent_w == 1] = a
                U_vs2[no_n0,i, h] += (np.average(P_step_vs[agent_e == 1], axis = 0) - c)/T
                Payments_vs2[no_n0,i,h] += np.average(P_step_vs, axis = 0)/T
            
            argsort_agent = P_once.argsort(axis = 0)
            for j, t in enumerate(Thresholds):
                t_prime = math.ceil(n*t)
                for k in range(5):
                    agent_w_t = np.zeros(n)
                    agent_w_t[argsort_agent[-t_prime:,k]] = 1
                    Acc_t[no_n0,i,j,k] += accuracy_computer_threshold(R, Y, agent_w_t)/T
                for h, a in enumerate(Amplitudes):
                    P_step_t = np.zeros((n,5)) + 0.1
                    P_step_t[argsort_agent[-t_prime:,k]] = a
                    U_t[no_n0,i, j, h] += (np.average(P_step_t[agent_e == 1], axis = 0) - c)/T
                    Payments_t[no_n0,i,j,h] += np.average(P_step_t, axis = 0)/T
                    
                    
                    
                    
                    