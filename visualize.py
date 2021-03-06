import matplotlib.pyplot as plt
import numpy as np
from random import *

fusion_techniques = ["bayesian", "ellipsoidal", "covariance intersection", "inverse covariance intersection", "probablistic consistent"]

def visualize(error, time, covariances, technique):
    home_dir_png = "results/images/"
    home_dir_pdf = "results/"
    
    directions = ["x", "y", "z"]
    
    n = randint(0, 700)
    num = n+100

    print(n)
    print(num)

    for i in range(3):
        fig = plt.figure()
        ax = plt.axes()
        plt.plot(time[n:num], error[i][n:num], color='blue', linewidth=0.5, label='True error')
        plt.plot(time[n:num], 3*covariances[i][n:num], color='red', linewidth=1, linestyle='dashed')
        plt.plot(time[n:num], -3*covariances[i][n:num], color='red', linewidth=1, linestyle='dashed', label='3 sigma')
        plt.xlabel("Time (s)")
        plt.ylabel("Positional Error "+ directions[i] +" (meters)")
        fig.set_figwidth(14)
        fig.tight_layout()
        ax.legend()
        plt.savefig(home_dir_png + technique +"_error("+ directions[i] +").png")
        plt.savefig(home_dir_pdf + technique +"_error("+ directions[i] +").pdf")
        plt.show()

if __name__ == '__main__':
    plt.cla()
    plt.clf()
    ax = plt.axes()
   
    for tech in fusion_techniques:
        errors = np.genfromtxt("results/errorCsv/"+tech+".csv",delimiter=',')
        mse = np.sum(np.square(errors), axis = 0)
        x = [ i for i in range(mse.size)]
        
        print(tech, np.average(mse))
        
        ax.plot(x, mse, label=tech)
    
    ax.legend()
    plt.show()
        