from data_structure import Data
import numpy as np
from utils import inv, normalize
from probConsistent import fusion


def getInitialInformationMatrix(data):
    I = np.zeros((4, 4))
    i = np.zeros((4, 1))
    for _ in range(4):
        H, y, R = data.getNextBatch()
        I = I + H.T @ H * inv(R) 
        i = i + H.T * inv(R) * y
    
    return I, i


def add_satellite(data, I, i):
    H, y, R = data.getNextBatch()

    #compute state dimensional from information 
    C = inv(I)
    mu = C @ i
    
    #normalize H first
    u = normalize(H)
    
    #project mat/vec in direction of u
    mu_a = u @ mu
    C_a = u @ C @ u.T
    
    #set-up b distribution
    mu_b = np.array(y).reshape(1, 1)
    C_b = np.array(R).reshape(1, 1)
    
    x_f, C_f = fusion(mu_a, mu_b, C_a, C_b, 1)
    
    #Compute additional information
    D = inv(inv(C_f) - inv(C_a))
    x_d = D @ (inv(C_f) @ x_f - inv(C_a) @ mu_a)
    
    
    fused_I = I + inv(D) * u.T @ u
    fused_i = i + u.T * inv(D) * x_d
    return fused_I, fused_i