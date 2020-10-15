import matplotlib.pyplot as plt
import numpy as np

fusion_techniques = ["bayesian", "ellipsoidal", "covariance intersection", "inverse covariance intersection", "probablistic consistent"]

fusion_colors = {
    "bayesian": 'grey',
    "ellipsoidal": 'red',
    "covariance intersection": 'blue',
    "inverse covariance intersection": 'purple',
    "probablistic consistent": 'green'
}
    

if __name__ == '__main__':
    plt.cla()
    plt.clf()
    fig = plt.figure()
    ax = plt.axes()
   
    for tech in fusion_techniques:
        std = np.genfromtxt("results/errorCsv/"+str(tech)+"_std_dev.csv",delimiter=',')
        x = [ i for i in range(std[0].size)]
        
        ax.plot(x, std[0], label=tech, color=fusion_colors[tech])
#         ax.plot(x, -std[0], color=fusion_colors[tech])
    
    ax.legend()
    fig.set_figwidth(14)
    fig.tight_layout()
    plt.show()
        