from data_input import DataLoader
from data_structure import Data
from computations import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from visualize import visualize
from data_fusion import Data_fusion

def pinv(H, y):
    return np.linalg.inv(H.T @ H) @ H.T @ y

fusion_techniques = ["bayesian", "ellipsoidal", "covariance intersection", "inverse covariance intersection", "probablistic consistent"]
if __name__ == '__main__':
    loader = DataLoader('data/data.csv')
    technique = int(input("Which fusion technique to use? (0 - bayesian, 1 - ei, 2 - ci, 3 - ici, 4 - pc)"))
    fusion = Data_fusion(technique)
    truth = loader.get_truth('data/truth.csv')
    anees = []
    estimations = np.zeros((6, loader.get_number_time_stamps()))
    for t in tqdm(range(loader.get_number_time_stamps())):
        data = loader.get_next_timestep()
        info_mat, info_vec = getInitialInformationMatrix(data)

        while not data.completed():
            info_mat, info_vec = add_satellite(data, info_mat, info_vec, fusion)
        pred = ((inv(info_mat) @ info_vec)/ data.getDeltaTime())[:3].reshape(3,)

        covs = np.sqrt(np.diag(inv(info_mat)))[:3].reshape(3,)
        estimations[:3,t] = pred
        estimations[3:,t] = covs
    
        error = pred[:3] - truth[:3,t]
        P_inv = inv(inv(info_mat)[:3,:3])
        anees.append(error @ P_inv @ error)

    
    
    print("ANEES:",np.average(anees))    
        
    time = truth[3]
    print(time.shape)
    
    error = estimations[:3] - truth[:3]
    
    covariances = estimations[3:]
    
    visualize(error, time, covariances, fusion_techniques[technique])
    
    np.savetxt("results/errorCsv/"+str(fusion_techniques[technique])+".csv", error, delimiter=",")
    np.savetxt("results/errorCsv/"+str(fusion_techniques[technique])+"_std_dev.csv", covariances, delimiter=",")
    
