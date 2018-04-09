# Homework 1: Classification using logistic regression
# Created on 04/08/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt

# Configurations
dimen = 2

#-----------------------------------------------------------------------------------------------------------------------
# Functions

def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    C = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        C = C +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, C, data_num)
    return ret_data
#-----------------------------------------------------------------------------------------------------------------------
# 1) Generating data
data_num = 200

# Class 0
theta0 = 0
m0 = np.zeros(2)
lamb0 = [2,1]
U0 = np.mat([[np.cos(theta0), np.sin(theta0)],[-np.sin(theta0), np.cos(theta0)]]).T

x1 = gaussianFromEigen(m0, lamb0, U0, data_num)

# Class 1
thetaA = -3*(np.pi)/4
pi_A = 1/3
mA = [-2,1]
lambA = [2,1/4]
UA = np.mat([[np.cos(thetaA), np.sin(thetaA)],[-np.sin(thetaA), np.cos(thetaA)]]).T

thetaB = (np.pi)/4
pi_B = 2/3
mB = [3,2]
lambB = [3,1]
UB = np.mat([[np.cos(thetaB), np.sin(thetaB)],[-np.sin(thetaB), np.cos(thetaB)]]).T

x2 = pi_A * gaussianFromEigen(mA, lambA, UA, data_num) + pi_B * gaussianFromEigen(mB, lambB, UB, data_num)

# Visualize data
plt.figure()
plt.scatter(x1[:,0],x1[:,1],s=5)
plt.scatter(x2[:,0],x2[:,1],s=5,c='r')
plt.title('Data visualization',fontsize=12)
plt.xlabel('Dimension 1',fontsize=10)
plt.ylabel('Dimension 2',fontsize=10)
ax1 = plt.gca()
ax1.set_aspect('equal', 'box')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# 2) Implement the MAP decision rule


#-----------------------------------------------------------------------------------------------------------------------
print('End')