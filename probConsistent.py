import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.linalg import sqrtm
from numpy.linalg import det
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from scipy.stats import invwishart as iw

def mutual_covariance(cov_a, cov_b):
    return max(cov_a, cov_b)

def get_critical_value(dimensions, alpha):
    return chi2.ppf((1 - alpha), df=dimensions)

def inv(mat):
    return np.linalg.inv(mat)

def compute_x_fus(x_a, x_b, C_a, C_b, C_c, C_f):    
    C_ac_inv = inv(C_a) - inv(C_c)
    C_bc_inv = inv(C_b) - inv(C_c)

    x_c = inv(inv(C_a) + inv(C_b) - 2*inv(C_c)) @ (inv(C_ac_inv) @ x_a + inv(C_bc_inv) @ x_b)
    
    x_ac = inv(C_ac_inv) @ (inv(C_a) @ x_a - inv(C_c) @ x_c)
    x_bc = inv(C_bc_inv) @ (inv(C_b) @ x_b - inv(C_c) @ x_c)
    
    return C_f @ (C_ac_inv @ x_ac + C_bc_inv @ x_bc + inv(C_c) @ x_c)

def fusion(x_a, x_b, C_a, C_b, dims):
    x_a = x_a.reshape(1, dims)
    x_b = x_b.reshape(1, dims)

    eta = get_critical_value(dims, 0.05)
    
    def objective(S):
        return -S[0]


    def prob_constraint(S):
        S = S.reshape(dims, dims).T
        C_c_inv = S@S.T

        C_ac = inv(inv(C_a) - C_c_inv)
        C_ac_inv = inv(C_ac)
        C_bc = inv(inv(C_b) - C_c_inv)
        C_bc_inv = inv(C_bc)

        C_abc_inv_inv = inv(C_ac_inv + C_bc_inv)
        C_abc_inv = inv(C_ac + C_bc)
        x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
        x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
        x_bc =(C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T
        f = ((x_ac - x_bc) @ C_abc_inv @ (x_ac - x_bc).T)[0][0]
        return eta - f

    cons = [{'type': 'eq', 'fun': prob_constraint}]
    S_0 = 0.99*(np.linalg.cholesky(inv(mutual_covariance(C_a, C_b))).T).reshape(dims**2, )

    sol = minimize(objective, S_0, method='SLSQP', constraints=cons)

    S = sol.x.reshape(dims, dims).T

    C_c_PC = inv(S.T) @ inv(S)
    
    C_fus = inv(inv(C_a) + inv(C_b) - inv(C_c_PC))
    x_fus = compute_x_fus(x_a, x_b, C_a, C_b, C_c_PC, C_fus)
        
    return x_fus, C_fus