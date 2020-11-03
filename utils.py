import numpy.linalg as LA

def inv(M):
    if M.size == 1:
        return M**(-1)
    else:
        return LA.inv(M)
    

def normalize(v):
    return v * (LA.norm(v))**(-1)