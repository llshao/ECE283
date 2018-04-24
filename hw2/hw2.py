# Homework 2: Classification using neural networks
# Created on 04/23/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt

# Configurations
dimen = 2 # data dimension

np.random.seed(0)
#-----------------------------------------------------------------------------------------------------------------------
# Functions
def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    Cov = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        Cov = Cov +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, Cov, data_num)
    return ret_data, Cov

# Implementation of scist.multivariate_normal.pdf(xi, mean_v, Cov)
def getGaussianLikelihood(x_i, mean_v, Cov):
    dimen = Cov.shape[0]
    GaussZ = np.power(2 * np.pi, (dimen * 0.5)) * np.power(np.linalg.det(Cov), 0.5)
    likelihood = np.exp(-0.5 * np.mat(x_i - mean_v) * (np.linalg.inv(Cov)) * (np.mat(x_i - mean_v).T)) / GaussZ
    return likelihood

# Evaluate binary classification result
def evalBCResult(pLabel, tLabel):
    incorrInd = np.squeeze(np.asarray((pLabel.flatten() != tLabel)))
    class0Ind = (tLabel == 0)
    class1Ind = (tLabel == 1)
    incorrPr0 = np.sum(incorrInd[class0Ind])/(class0Ind.shape[0])
    incorrPr1 = np.sum(incorrInd[class1Ind])/(class1Ind.shape[0])
    print("Class 0 - error = {0:4.1f}%\nClass 1 - error = {1:4.1f}%\n".format(100*incorrPr0,100*incorrPr1))
    return incorrInd

#-----------------------------------------------------------------------------------------------------------------------
# Generating data
default_data_num = 400

# Class 0
theta0 = 0
m0 = np.zeros(dimen)
lamb0 = [2,1]
U0 = np.mat([[np.cos(theta0), np.sin(theta0)],[-np.sin(theta0), np.cos(theta0)]]).T

x0, C0 = gaussianFromEigen(m0, lamb0, U0, default_data_num)

# Class 1
thetaA = -3*(np.pi)/4
pi_A = 1/3
mA = np.array([-2,1])
lambA = [2,1/4]
UA = np.mat([[np.cos(thetaA), np.sin(thetaA)],[-np.sin(thetaA), np.cos(thetaA)]]).T

thetaB = (np.pi)/4
pi_B = 2/3
mB = np.array([3,2])
lambB = [3,1]
UB = np.mat([[np.cos(thetaB), np.sin(thetaB)],[-np.sin(thetaB), np.cos(thetaB)]]).T

x1A, C1A = gaussianFromEigen(mA, lambA, UA, default_data_num)
x1B, C1B = gaussianFromEigen(mB, lambB, UB, default_data_num)
mixGaussPi = np.random.uniform(0.0,1.0,default_data_num)

x1 = np.concatenate((x1A[mixGaussPi <= pi_A,:],x1B[mixGaussPi > pi_A,:]),axis=0)
x1 = x1[np.random.permutation(default_data_num),:] # Reshuffle the gaussian mixture data

print('C0 =\n',C0, '\n C1A =\n',C1A,'\n C1B =\n',C1B)

trainEnd = int(default_data_num*0.7)
testEnd = int(default_data_num*0.9)

xTrain = np.concatenate((x0[:trainEnd,:],x1[:trainEnd,:]),axis=0) # data
tTrain = np.concatenate((np.zeros(trainEnd),np.ones(trainEnd))) # label

xTest = np.concatenate((x0[trainEnd:testEnd,:],x1[trainEnd:testEnd,:]),axis=0) # data
tTest = np.concatenate((np.zeros(testEnd-trainEnd),np.ones(testEnd-trainEnd))) # label

xValid = np.concatenate((x0[testEnd:,:],x1[testEnd:,:]),axis=0) # data
tValid = np.concatenate((np.zeros(default_data_num-testEnd),np.ones(default_data_num-testEnd))) # label

# Shuffle the training set
randInd = np.random.permutation(xTrain.shape[0])
xTrain = xTrain[randInd,:]
tTrain = tTrain[randInd]

# # Visualize data
# plt.figure()
# plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
# plt.scatter(xTrain[(tTrain==0),0],xTrain[(tTrain==0),1],s=5,label='Train (Class 0)')
# plt.scatter(xTrain[(tTrain==1),0],xTrain[(tTrain==1),1],s=5,c='r',label='Train (Class 1)')
# plt.scatter(xTest[(tTest==0),0],xTest[(tTest==0),1],s=5,c='g',label='Test (Class 0)')
# plt.scatter(xTest[(tTest==1),0],xTest[(tTest==1),1],s=5,c='c',label='Test (Class 1)')
# plt.scatter(xValid[(tValid==0),0],xValid[(tValid==0),1],s=5,c='k',label='Valid (Class 0)')
# plt.scatter(xValid[(tValid==1),0],xValid[(tValid==1),1],s=5,c='m',label='Valid (Class 1)')
# plt.title('Data visulization',fontsize=12)
# plt.xlabel('Dimension 0',fontsize=10)
# plt.ylabel('Dimension 1',fontsize=10)
# plt_ax = plt.gca()
# plt_ax.set_aspect('equal', 'box')
# plt_ax.legend(loc='upper left',bbox_to_anchor=(0.1, 1.0))
# plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# 1) implement a fully connected neural network
# (a) 1 hidden layer

# (b) 2 hidden layer



#-----------------------------------------------------------------------------------------------------------------------
print('End')