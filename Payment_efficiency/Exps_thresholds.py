# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:06:40 2021

@author: yichizhang
"""

from Model_And_Mechanisms import mechanism_matrix_fast, mechanism_determinant_fast, mechanism_pairing, Report_Generator, Crowdsourcing
from Model_And_Mechanisms import mechanism_determinant_gt, mechanism_matrix_gt, mechanism_matrix_det_2
import numpy as np
import math
import matplotlib.pyplot as plt

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
#%%
"""
Experiment: comparing optimal step, virtual shirker step, and iterated virtual shirker step
For matrix and determinant
"""
m = 100
mi = 50
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
N0 = np.arange(10, 32, 2)
T = 80
alpha = 5
k_max = 20
U_vs = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
U_vs2 = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
U_wmv = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
# U_t = np.zeros((len(N0), len(Effort), len(Thresholds), len(Amplitudes), 5))
Payments_vs = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
Payments_vs2 = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
Payments_wmv = np.zeros((len(N0), len(Effort), len(Amplitudes), 5))
# Payments_t = np.zeros((len(N0), len(Effort), len(Thresholds), len(Amplitudes), 5))
Acc_vs = np.zeros((len(N0), len(Effort), 5))
Acc_vs2 = np.zeros((len(N0), len(Effort), 5))
Acc_wmv = np.zeros((len(N0), len(Effort), 5))
# Acc_t = np.zeros((len(N0), len(Effort), len(Thresholds), 5))

for no_n0, n0 in enumerate(N0):
    n = math.ceil(m*n0/(mi - 1))
    for l in range(T):
        if l%10 == 0:
            print('n0 ',n0,' ; round ',l)
        for i, e in enumerate(Effort):
            para_w = Crowdsourcing(w = w, Gamma_w = Gamma_w, Gamma_s = Gamma_s, e = e, mi = mi, n0 = n0)
            para_s = Crowdsourcing(w = w, Gamma_w = Gamma_w, Gamma_s = Gamma_random, prior_e = [1,0,0], mi = mi, n0 = int(n0/5))
            R, Y, agent_e = Report_Generator(para_w)
            agent_w = np.ones((n,5))
            agent_w_once = np.ones((n,5))
            P_once = np.zeros((n,5))
            count = 0
            while True:
                agent_w_new = agent_w.copy()
                for k in range(5):
                    
                    n_w = np.count_nonzero(agent_w[:,k])
                    #n_s = int(n_w/5) + 1
                    R_s, _, _ = Report_Generator(para_s)
                    n_s = np.size(R_s, axis = 0)
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
            
            R_s, _, _ = Report_Generator(para_s)
            c = cost(e, alpha)
            agent_w_wmj = np.ones((n,5))
            for k in range(5):
                Acc_vs[no_n0,i,k] += accuracy_computer_threshold(R, Y, agent_w_once[:,k])/T
                Acc_vs2[no_n0,i,k] += accuracy_computer_threshold(R, Y, agent_w[:,k])/T
                acc_wmv, agent_wmj = iterated_weighted_maj(R, R_s, Y, k, k_max, para_w)
                Acc_wmv[no_n0,i,k] += acc_wmv/T
                agent_w_wmj[:,k] = agent_wmj[0:n]
                
            for h, a in enumerate(Amplitudes):
                P_step_vs = np.zeros((n,5)) + 0.1
                P_step_vs[agent_w_once == 1] = a
                U_vs[no_n0,i, h] += (np.average(P_step_vs[agent_e == 1], axis = 0) - c)/T
                Payments_vs[no_n0,i,h] += np.average(P_step_vs, axis = 0)/T
                
                P_step_vs = np.zeros((n,5)) + 0.1
                P_step_vs[agent_w == 1] = a
                # if h == 4:
                #     print(e, a)
                #     print(c)
                #     print((np.average(P_step_vs[agent_e == 1], axis = 0) - c), '\n')
                uti = (np.average(P_step_vs[agent_e == 1], axis = 0) - c)
                if uti.any() > 0 and a < c:
                    print('errrrrrrrrrrrrrr')
                U_vs2[no_n0,i, h] += uti/T
                Payments_vs2[no_n0,i,h] += np.average(P_step_vs, axis = 0)/T
                
                P_step_vs = np.zeros((n,5)) + 0.1
                P_step_vs[agent_w_wmj == 1] = a
                U_wmv[no_n0,i, h] += (np.average(P_step_vs[agent_e == 1], axis = 0) - c)/T
                Payments_wmv[no_n0,i,h] += np.average(P_step_vs, axis = 0)/T
                
            
            # argsort_agent = P_once.argsort(axis = 0)
            # for j, t in enumerate(Thresholds):
            #     t_prime = math.ceil(n*t)
            #     for k in range(5):
            #         agent_w_t = np.zeros(n)
            #         agent_w_t[argsort_agent[-t_prime:,k]] = 1
            #         Acc_t[no_n0,i,j,k] += accuracy_computer_threshold(R, Y, agent_w_t)/T
            #     for h, a in enumerate(Amplitudes):
            #         P_step_t = np.zeros((n,5)) + 0.1
            #         P_step_t[argsort_agent[-t_prime:,k]] = a
            #         U_t[no_n0,i, j, h] += (np.average(P_step_t[agent_e == 1], axis = 0) - c)/T
            #         Payments_t[no_n0,i,j,h] += np.average(P_step_t, axis = 0)/T
                    

#%%
"""
Data analysis
"""
def min_payment(Acc_goal, no_n0):
    Payment_min_vs = np.zeros(5)
    Effort_min_vs = np.zeros(5)
    Payment_min_vs2 = np.zeros(5)
    Effort_min_vs2 = np.zeros(5)
    Payment_min_wmv = np.zeros(5)
    Effort_min_wmv = np.zeros(5)

    Effort_emp = np.argmax(U_vs[no_n0], axis = 0)
    for k in range(5):
        if np.count_nonzero(Acc_vs[no_n0,:,k] >= Acc_goal) == 0:
            amplitude_range_vs = []
        else:
            index_acc = next(x for x in Acc_vs[no_n0,:,k] if x >= Acc_goal)
            min_effort_vs = np.where(Acc_vs[no_n0,:,k] == index_acc)[0][0]
            amplitude_range_vs = np.where(Effort_emp[:,k] >= min_effort_vs)[0]
        pay_min_k = np.inf
        eff_min_k = -np.inf
        for h in amplitude_range_vs:
            if Payments_vs[no_n0,Effort_emp[h,k],h,k] < pay_min_k:
                pay_min_k = Payments_vs[no_n0,Effort_emp[h,k],h,k]
                eff_min_k = Effort[Effort_emp[h,k]]
        Payment_min_vs[k] = pay_min_k
        Effort_min_vs[k] = eff_min_k
    
    Effort_emp = np.argmax(U_vs2[no_n0], axis = 0)
    for k in range(5):
        if np.count_nonzero(Acc_vs2[no_n0,:,k] >= Acc_goal) == 0:
            amplitude_range_vs = []
        else:
            index_acc = next(x for x in Acc_vs2[no_n0,:,k] if x >= Acc_goal)
            min_effort_vs = np.where(Acc_vs2[no_n0,:,k] == index_acc)[0][0]
            amplitude_range_vs = np.where(Effort_emp[:,k] >= min_effort_vs)[0]
        pay_min_k = np.inf
        eff_min_k = -np.inf
        amp = 0
        for h in amplitude_range_vs:
            if Payments_vs2[no_n0,Effort_emp[h,k],h,k] < pay_min_k:
                pay_min_k = Payments_vs2[no_n0,Effort_emp[h,k],h,k]
                eff_min_k = Effort[Effort_emp[h,k]]
                amp = h
        Payment_min_vs2[k] = pay_min_k
        Effort_min_vs2[k] = eff_min_k
        print('opt_amp: ', Amplitudes[amp], k, no_n0)
        print('elicited effort: ', eff_min_k)
        print('cost of effort: ', cost(eff_min_k, a))
        
    Effort_emp = np.argmax(U_wmv[no_n0], axis = 0)
    for k in range(5):
        if np.count_nonzero(Acc_wmv[no_n0,:,k] >= Acc_goal) == 0:
            amplitude_range_vs = []
        else:
            index_acc = next(x for x in Acc_wmv[no_n0,:,k] if x >= Acc_goal)
            min_effort_vs = np.where(Acc_wmv[no_n0,:,k] == index_acc)[0][0]
            amplitude_range_vs = np.where(Effort_emp[:,k] >= min_effort_vs)[0]
        pay_min_k = np.inf
        eff_min_k = -np.inf
        for h in amplitude_range_vs:
            if Payments_wmv[no_n0,Effort_emp[h,k],h,k] < pay_min_k:
                pay_min_k = Payments_wmv[no_n0,Effort_emp[h,k],h,k]
                eff_min_k = Effort[Effort_emp[h,k]]
        Payment_min_wmv[k] = pay_min_k
        Effort_min_wmv[k] = eff_min_k

    # Payment_min_opt = np.zeros(5)
    # Effort_min_opt = np.zeros(5)
    
    # Effort_opt = np.argmax(U_t[no_n0], axis = 0)
    # for k in range(5):
    #     pay_min_k = np.inf
    #     eff_min_k = -np.inf
    #     for j,t in enumerate(Thresholds):
    #         if np.count_nonzero(Acc_t[no_n0,:,j,k] >= Acc_goal) == 0:
    #             amplitude_range_t = []
    #         else:
    #             index_acc = next(x for x in Acc_t[no_n0,:,j,k] if x >= Acc_goal)
    #             min_effort_t = np.where(Acc_t[no_n0,:,j,k] == index_acc)[0][0]
    #             amplitude_range_t = np.where(Effort_opt[j,:,k] >= min_effort_t)[0]
          
    #         for h in amplitude_range_t:
    #             if Payments_t[no_n0,Effort_opt[j,h,k],j,h,k] < pay_min_k:
    #                 pay_min_k = Payments_t[no_n0,Effort_opt[j,h,k],j,h,k]
    #                 eff_min_k = Effort[Effort_opt[j,h,k]]
    #     Payment_min_opt[k] = pay_min_k
    #     Effort_min_opt[k] = eff_min_k
    return Payment_min_vs, Payment_min_vs2, Payment_min_wmv, Effort_min_vs, Effort_min_vs2, Effort_min_wmv

acc = 0.99
MI = ['Matrix-TVD', 'Matrix-KL', 'Matrix-SQ', 'Matrix-HLG', 'DMI']
minpay_vs2 = np.zeros((len(N0), 5))
mineffort_vs2 = np.zeros((len(N0), 5))
for no_n0 in range(len(N0)):
    Payment_min_vs, Payment_min_vs2, Payment_min_opt, Effort_min_vs, Effort_min_vs2, Effort_min_opt = min_payment(acc, no_n0)
    minpay_vs2[no_n0] = Payment_min_vs2*math.ceil(m*N0[no_n0]/(mi - 1))
    mineffort_vs2[no_n0] = Effort_min_vs2

# plt.figure()
# for k in range(5):
#     plt.plot(N0, minpay_vs2[:,k], label = MI[k])
# plt.xlabel('Number of agents assigned to each task')
# plt.ylabel('Minpayment')
# plt.title('Iterated vs, World 1, Matrix and DMI, mi = '+str(mi)+', Acc = '+str(acc))
# plt.legend()
    
# plt.figure()
# for k in range(5):
#     plt.plot(N0, mineffort_vs2[:,k], label = MI[k])
# plt.xlabel('Number of agents assigned to each task')
# plt.ylabel('Elicited effort')
# plt.title('Iterated vs, World 1, Matrix and DMI, mi = '+str(mi)+', Acc = '+str(acc))
# plt.legend()

#%%
Paymin = np.zeros((5, len(N0), 3))
for no_n0, n0 in enumerate(N0):
    Payment_min_vs, Payment_min_vs2, Payment_min_wmv, _, _, _ = min_payment(acc, no_n0)
    for k in range(5):
        Paymin[k, no_n0, 0] = Payment_min_vs[k]
        Paymin[k, no_n0, 1] = Payment_min_vs2[k]
        Paymin[k, no_n0, 2] = Payment_min_wmv[k]
    
MI = ['Matrix-TVD', 'Matrix-KL', 'Matrix-SQ', 'Matrix-HLG', 'DMI']
for k in range(5):
    plt.figure()
    plt.plot(N0, Paymin[k,:,0], label = 'VS')
    plt.plot(N0, Paymin[k,:,1], label = 'Iterated-VS')
    plt.plot(N0, Paymin[k,:,2], label = 'Weighted Majority Vote')
    plt.xlabel('Goal accuracy')
    plt.ylabel('Minimum payment')
    plt.title('World_1-mi_30-m_100-ct_100, '+ MI[k])
    plt.legend()

#%%
"""
Error estimation
"""
def classifier_error_matrix(R, para_s):
    n = np.size(R, axis = 0)
    n_3 = int(n/3)
    R_A = R[0:n_3]
    R_B = R[n_3:n_3*2]
    R_C = R[n_3*2:n]
    e_AB = mechanism_matrix_det_2(R_A, R_B, para_s)
    e_AC = mechanism_matrix_det_2(R_A, R_C, para_s)
    
    err = np.zeros(5)
    for j in range(5):
        n_s_hat = np.size(np.intersect1d(np.where(e_AB[:,j] == 0), np.where(e_AC[:,j] == 0)))
        n_s_B = np.size(np.where(e_AB[:,j] == 0))
        n_s_C = np.size(np.where(e_AC[:,j] == 0))
        err[j] = (n_s_B+n_s_C-2*n_s_hat)/(2*n_3)
    return err

n0 = 20
mi = 60
e = 0.7
para_w = Crowdsourcing(w = w, Gamma_w = Gamma_w, Gamma_s = Gamma_s, e = e, mi = mi, n0 = n0)
para_s = Crowdsourcing(w = w, Gamma_w = Gamma_w, Gamma_s = Gamma_random, prior_e = [1,0,0], mi = mi, n0 = int(n0/10))
R, Y, agent_e = Report_Generator(para_w)
print(classifier_error_matrix(R, para_s))



#%%
print(U_vs2[4, :, 4, 0])
print(Amplitudes[4])
print(Effort[np.argmax(U_vs2[4, :, 4, 0])])
print(cost(Effort[np.argmax(U_vs2[4, :, 4, 0])], a))
#%%
P_step_vs = np.zeros((n,5)) + 0.1
P_step_vs[agent_w == 1] = a
print(P_step_vs)
U_vs2[no_n0,i, h] += (np.average(P_step_vs[agent_e == 1], axis = 0) - c)















            
                    
                    
                    