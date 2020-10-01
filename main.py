from data_input import DataLoader
from data_structure import Data
from computations import *
import numpy as np

def pinv(H, y):
    return np.linalg.inv(H.T @ H) @ H.T @ y

if __name__ == '__main__':
    loader = DataLoader('data.csv')
    
    
    for _ in range(10):
        data = loader.get_next_timestep()
        info_mat, info_vec = getInitialInformationMatrix(data)

        while not data.completed():
            info_mat, info_vec = add_satellite(data, info_mat, info_vec)
        pred = (inv(info_mat) @ info_vec)/ data.getDeltaTime()
        print(pred)
        print("==============================")
    