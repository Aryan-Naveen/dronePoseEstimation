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
from constraints import Constraints

def inv(A):
    return LA.inv(A)


def mutual_covariance(cov_a, cov_b):
    D_a, S_a = np.linalg.eigh(cov_a)
    D_a_sqrt = sqrtm(np.diag(D_a))
    D_a_sqrt_inv = inv(D_a_sqrt)
    M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
    D_b, S_b = np.linalg.eigh(M)
    D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
    return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a in Sijs et al

def get_critical_value(dimensions, alpha):
    return chi2.ppf((1 - alpha), df=dimensions)

def fusion(x_a, x_b, C_a, C_b, gamma):
    eta = get_critical_value(2, gamma)
    S_0 = np.array([0])
    constraints = Constraints(eta, 1)
    cons = [{'type': 'eq', 'fun': constraints.prob_constraint}]
    
    constraints.update_eta(get_critical_value(2, gamma))
    sol = minimize(constraints.objective, S_0, method='SLSQP', constraints=cons, tol=1e-5)

    G = sol.x
    C_c_EI =  mutual_covariance(C_a, C_b)
    C_c_PC_05 = C_c_EI + G @ G.T
    fus_PC_05 = inv(inv(C_a) + inv(C_b) - inv(C_c_PC_05))

    