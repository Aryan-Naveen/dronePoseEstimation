from scipy.optimize import fminbound
from numpy.linalg import inv, det
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import chi2
import numpy.linalg as LA


def get_critical_value(dimensions, alpha):
    return chi2.ppf((1 - alpha), df=dimensions)

class Data_fusion():
    def __init__(self, t):
        fusion_approaches = {0: self._bayesian_fusion,
                             1: self._ei_fusion,
                             2: self._covariance_fusion,
                             3: self._ici_fusion,
                             4: self._pc_fusion}
        self.fusion_func=fusion_approaches[t]
    
    def _covariance_fusion(self, mean_a, mean_b, cov_a, cov_b):
        def optimize_omega(cov_a, cov_b):
            return 0.5

        omega = optimize_omega(cov_a, cov_b)
        cov = inv(np.multiply(omega, inv(cov_a)) + np.multiply(1 - omega, inv(cov_b)))
        mean = np.dot(cov, (np.dot(np.multiply(omega, inv(cov_a)), mean_a) + np.dot(np.multiply(1 - omega, inv(cov_b)), mean_b)))
        return mean, cov
            
    def _bayesian_fusion(self, mu_a, mu_b, C_a, C_b):
        C_f = np.linalg.inv(np.linalg.inv(C_a) + np.linalg.inv(C_b))
        x_f = C_f @ (np.linalg.inv(C_a) @ mu_a + np.linalg.inv(C_b) @ mu_b)
        return x_f, C_f

    def _ei_fusion(self, x_a, x_b, C_a, C_b):
        C_c = max(C_a, C_b) + 1e-10*np.identity(1)

        C_ac = inv(inv(C_a) - inv(C_c))
        C_ac_inv = inv(C_ac)
        C_bc = inv(inv(C_b) - inv(C_c))
        C_bc_inv = inv(C_bc)

        C_abc_inv_inv = inv(C_ac_inv + C_bc_inv)
        C_abc_inv = inv(C_ac + C_bc)
        x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
        x_ac = (C_ac @ (inv(C_a) @ x_a.T - inv(C_c) @ x_c.T)).T
        x_bc =(C_bc @ (inv(C_b) @ x_b.T - inv(C_c) @ x_c.T)).T

        C_fus = inv(inv(C_a) + inv(C_b) - inv(C_c))
        x_fus = C_fus @ (C_ac_inv @ x_ac + C_bc_inv @ x_bc + inv(C_c) @ x_c)

        return x_fus, C_fus
    
    def _ici_fusion(self, mean_a, mean_b, cov_a, cov_b):
        def optimize_omega(cov_a, cov_b):
            def optimize_fn(omega):
                return det(inv(inv(cov_a) + inv(cov_b) - inv(np.multiply(omega, cov_a) + np.multiply(1 - omega, cov_b))))
            return fminbound(optimize_fn, 0, 1)
        omega = optimize_omega(cov_a, cov_b)
        cov = inv(inv(cov_a) + inv(cov_b) - inv(np.multiply(omega, cov_a) + np.multiply(1 - omega, cov_b)))
        T = inv(np.multiply(omega, cov_a) + np.multiply(1-omega, cov_b))
        K = np.dot(cov, inv(cov_a) - np.multiply(omega, T))
        L = np.dot(cov, inv(cov_b) - np.multiply(1 - omega, T))
        mean = np.dot(K, mean_a) + np.dot(L, mean_b)
        return mean, cov


    def _pc_fusion(self, x_a, x_b, C_a, C_b):
        eta = get_critical_value(1, 0.05)
        def prob_constraint(C_c_inv):

            C_ac = inv(inv(C_a) - C_c_inv)
            C_bc = inv(inv(C_b) - C_c_inv)

            C_abc_inv_inv = inv(inv(C_ac) + inv(C_bc))
            C_abc_inv = inv(C_ac + C_bc)

            x_c = (C_abc_inv_inv @ (LA.inv(C_ac) @ x_a.T + LA.inv(C_bc) @ x_b.T)).T

            x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
            x_bc = (C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T

            f = ((x_ac - x_bc) @ LA.inv(C_ac+C_bc) @ (x_ac - x_bc).T)[0][0]
            return (eta-f)**2
        
        C_c = inv(fminbound(prob_constraint, 0, inv(max(C_a, C_b) + 1e-25*np.identity(1))))
        C_ac = inv(inv(C_a) - inv(C_c))
        C_ac_inv = inv(C_ac)
        C_bc = inv(inv(C_b) - inv(C_c))
        C_bc_inv = inv(C_bc)

        C_abc_inv_inv = inv(C_ac_inv + C_bc_inv)
        C_abc_inv = inv(C_ac + C_bc)
        x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
        x_ac = (C_ac @ (inv(C_a) @ x_a.T - inv(C_c) @ x_c.T)).T
        x_bc =(C_bc @ (inv(C_b) @ x_b.T - inv(C_c) @ x_c.T)).T

        C_fus = inv(inv(C_a) + inv(C_b) - inv(C_c))
        x_fus = C_fus @ (C_ac_inv @ x_ac + C_bc_inv @ x_bc + inv(C_c) @ x_c)

        return x_fus, C_fus

    
    def fuse(self, x_a, x_b, C_a, C_b):
        return self.fusion_func(x_a, x_b, C_a, C_b)
        
        