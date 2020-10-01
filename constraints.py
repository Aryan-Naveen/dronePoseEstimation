import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.linalg import sqrtm
from numpy.linalg import det
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import numpy as np
from scipy.stats import invwishart as iw        
import matplotlib.pyplot as plt

def inv(A):
    return LA.inv(A)


def mutual_covariance(cov_a, cov_b):
    D_a, S_a = np.linalg.eigh(cov_a)
    D_a_sqrt = sqrtm(np.diag(D_a))
    D_a_sqrt_inv = inv(D_a_sqrt)
    M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
    D_b, S_b = np.linalg.eigh(M)
    D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
    return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a 

def pinv(A):
    RELU = np.vectorize(relu)
    tmp_eig, tmp_egv = LA.eig(A)
    M_inv = tmp_egv @ np.diag(1/RELU(tmp_eig)) @ tmp_egv.T
    M = tmp_egv @ np.diag(RELU(tmp_eig)) @ tmp_egv.T
    return M_inv

def relu(v):
#     threshold = 1E-5
#     if v < threshold:
#         return np.log1p(1 + np.exp(v))* threshold /np.log1p(1+np.exp(threshold))
#     else:
#         return v
    return max(v, 1E-5)



class Constraints():
    def __init__(self, eta, dims):
        self.e = eta
        self.d = dims 
        x_a = mu_a
        C_a = P_a
        x_b = mu_b
        C_b = P_b
    
    def update_eta(self, new_eta):
        self.e=new_eta
    
    def objective(self, S):
        G = np.zeros((self.d, self.d))
        G[np.tril_indices(self.d)] = S
        return np.linalg.det(G @ G.T + 1e-10*np.identity(self.d))

    def constraint1(self, S):
        G = np.zeros((self.d, self.d))
        G[np.tril_indices(self.d)] = S
        A = G@G.T - 1e-10*np.identity(self.d)
        return np.linalg.eigh(A)[0][0]

    def prob_constraint(self, S):
        G = np.zeros((self.d, self.d))
        G[np.tril_indices(self.d)] = S
        C_c_inv = LA.inv(mutual_covariance(C_a, C_b) + 1e-10*np.identity(self.d) + G@G.T)

        C_ac = pinv(inv(C_a) - C_c_inv)
        C_bc = pinv(inv(C_b) - C_c_inv)

        C_abc_inv_inv = pinv(pinv(C_ac) + pinv(C_bc))
        C_abc_inv = pinv(C_ac + C_bc)

        x_c = (C_abc_inv_inv @ (LA.inv(C_ac) @ x_a.T + LA.inv(C_bc) @ x_b.T)).T

        x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
        x_bc = (C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T

        f = ((x_ac - x_bc) @ LA.inv(C_ac+C_bc) @ (x_ac - x_bc).T)[0][0]
        return self.e-f

    def debug(self, S):
        print(S)
        print ('objective is',self.objective(S))
        print ('constraint1 is ',self.constraint1(S))
        print ('prob_constraint is ',self.prob_constraint(S))