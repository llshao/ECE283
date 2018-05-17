# Homework 4: Classification using SVM and Adaboost
# Created on 05/16/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import contextlib
import numpy as np
import numpy.matlib as matlib
import numpy.linalg as nla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC

#-----------------------------------------------------------------------------------------------------------------------
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
data_num = 300

# Class 0
theta0 = 0
m0 = np.zeros(dimen)
lamb0 = [2,1]
U0 = np.mat([[np.cos(theta0), np.sin(theta0)],[-np.sin(theta0), np.cos(theta0)]]).T

x0, C0 = gaussianFromEigen(m0, lamb0, U0, data_num)

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

x1A, C1A = gaussianFromEigen(mA, lambA, UA, data_num)
x1B, C1B = gaussianFromEigen(mB, lambB, UB, data_num)
mixGaussPi = np.random.uniform(0.0,1.0,data_num)

x1 = np.concatenate((x1A[mixGaussPi <= pi_A,:],x1B[mixGaussPi > pi_A,:]),axis=0)
x1 = x1[np.random.permutation(data_num),:] # Reshuffle the gaussian mixture data

print('C0 =\n',C0, '\n C1A =\n',C1A,'\n C1B =\n',C1B)

trainEnd = int(data_num*0.6666)

xTrain = np.concatenate((x0[:trainEnd,:],x1[:trainEnd,:]),axis=0) # data
tTrain = np.concatenate((np.zeros(trainEnd),np.ones(trainEnd))) # label

xTest = np.concatenate((x0[trainEnd:,:],x1[trainEnd:,:]),axis=0) # data
tTest = np.concatenate((np.zeros(data_num-trainEnd),np.ones(data_num-trainEnd))) # label

# Shuffle the training set
trainLen = xTrain.shape[0]
randInd = np.random.permutation(trainLen)
xTrain = xTrain[randInd,:]
tTrain = tTrain[randInd]

#-----------------------------------------------------------------------------------------------------------------------
# 1) SVM-SMO
clf = SVC()
clf.fit(xTrain, tTrain)
tPred = clf.predict(xTest)
evalBCResult(tPred, tTest)
