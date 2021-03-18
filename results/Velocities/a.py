from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
from random import *
class ResultsData():
    def __init__(self):
        self.truth = genfromtxt('truth.csv', delimiter=',')
        self.bayesian = genfromtxt('bayesian_computed_vel.csv', delimiter=',')
        self.ci = genfromtxt('covariance intersection_computed_vel.csv', delimiter=',')
        self.ei = genfromtxt('ellipsoidal_computed_vel.csv', delimiter=',')
        self.pc = genfromtxt('probablistic consistent_computed_vel.csv', delimiter=',')
        self.times = genfromtxt('time.csv', delimiter=',')

    def getDeltaTime(self, timestep):
        return self.times[timestep][0] - self.times[timestep][1]
    
    def get_bayesian_positions(self):
        poses = []
        pose = np.array([0, 0, 0])
        t = 0
        U = []
        V = []
        for t in range(int(self.bayesian.size/3)):
            vel = self.bayesian[:,t]
            U.append(vel[0])
            V.append(vel[1])
            pose = pose + vel*self.getDeltaTime(t)
            poses.append(pose)
        
        poses = np.array(poses)
        return poses, U, V
    
    
    def get_true_positions(self):
        poses = []
        pose = np.array([0, 0, 0])
        t = 0
        U = []
        V = []
        Z = []
        for t in range(int(self.truth.size/3) - 1):
            vel = self.truth[:,t]
            U.append(vel[0])
            V.append(vel[1])
            Z.append(vel[2])
            pose = pose + vel*self.getDeltaTime(t)
            poses.append(pose)
        
        poses = np.array(poses)
        return poses, U, V, Z
    
    def get_pc_positions(self):
        poses = []
        pose = np.array([0, 0, 0])
        t = 0
        U = []
        V = []
        Z = []
        for t in range(int(self.pc.size/3)):
            vel = self.pc[:,t]
            U.append(vel[0])
            V.append(vel[1])
            Z.append(vel[2])
            pose = pose + vel*self.getDeltaTime(t)
            poses.append(pose)
        
        poses = np.array(poses)
        return poses, U, V, Z
    
    
    def get_ci_positions(self):
        poses = []
        pose = np.array([0, 0, 0])
        t = 0
        U = []
        V = []
        Z = []
        for t in range(int(self.ci.size/3)):
            vel = self.ci[:,t]
            U.append(vel[0])
            V.append(vel[1])
            Z.append(vel[2])
            pose = pose + vel*self.getDeltaTime(t)
            poses.append(pose)
        
        poses = np.array(poses)
        return poses, U, V, Z


a = 0
b = a + 35

l = 1
results = ResultsData()
fig = plt.figure()
ax = fig.gca(projection='3d')


poses, U, V, Z = results.get_true_positions()
x = poses[:, 0]
y = poses[:, 1]
z = poses[:, 2]
ax.plot(x[a:b], y[a:b], z[a:b], color='black', label='truth')

poses, U, V, Z = results.get_pc_positions()
x = poses[:, 0]
y = poses[:, 1]
z = poses[:, 2]
ax.plot(x[a:b], y[a:b], z[a:b], color='red', label='PC')

poses, U, V, Z = results.get_ci_positions()
x = poses[:, 0]
y = poses[:, 1]
z = poses[:, 2]
ax.plot(x[a:b], y[a:b], z[a:b], color='blue', label='CI')

print(a)
print(b)
# poses, U, V = results.get_pc_positions()
# x = poses[:, 0]
# y = poses[:, 1]
# plt.quiver(x[a:b], y[a:b], U[a:b], V[a:b], color = 'blue', label='PC', linewidth=l)

# poses, U, V = results.get_ci_positions()
# x = poses[:, 0]
# y = poses[:, 1]
# plt.quiver(x[a:b], y[a:b], U[a:b], V[a:b], color = 'green', label='CI', linewidth=l)


# poses, U, V = results.get_bayesian_positions()
# x = poses[:, 0]
# y = poses[:, 1]
# plt.quiver(x[a:b], y[a:b], U[a:b], V[a:b], color = 'red', label='bayesian', linewidth=l)
# print(V[a:b])

plt.show()
    
