# Homework 1: Classification using logistic regression
# Created on 04/08/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import scipy.stats as scist

# Configurations
dimen = 2

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

# Implement the MAP decision rule
def binaryMAP(data_in):
    classPr = np.empty(shape=[0, dimen])
    for xi in data_in:
        Pr_t0 = getGaussianLikelihood(xi, m0, C0)

        Pr_t1A = getGaussianLikelihood(xi, mA, C1A)
        Pr_t1B = getGaussianLikelihood(xi, mB, C1B)
        Pr_t1 = pi_A * Pr_t1A + pi_B * Pr_t1B

        classPr = np.append(classPr,np.concatenate((Pr_t0,Pr_t1),axis=1),axis=0)

    return classPr

# Construct 10-dimentional feature vector
def nonKer10Feature(data_in):
    Phi = np.ones((10, data_in.shape[0]))
    Phi[1, :] = data_in[:, 0]
    Phi[2, :] = data_in[:, 1]
    Phi[3, :] = np.square(data_in[:, 0])
    Phi[4, :] = np.square(data_in[:, 1])
    Phi[5, :] = np.multiply(data_in[:, 0], data_in[:, 1])
    Phi[6, :] = np.power(data_in[:, 0], 3)
    Phi[7, :] = np.multiply(np.square(data_in[:, 0]), data_in[:, 1])
    Phi[8, :] = np.multiply(data_in[:, 0], np.square(data_in[:, 1]))
    Phi[9, :] = np.power(data_in[:, 1], 3)
    return Phi

# logistic sigmoid functio
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#  Predict using a (non-kernelized) logistic regression binary classifier
def predictLRBC(Phi,w):
    y = sigmoid(w.dot(Phi))
    return (y > 0.5).astype(int)

#  Training a (non-kernelized) logistic regression binary classifier
def trainLRBC(Phi, tlabel, maxIter = 100, wToler = 2e-16):
    featureN = Phi.shape[0] # dimension of features (must be rows of Phi)
    w = np.zeros(featureN)
    for i in range(maxIter):
        y = sigmoid(w.dot(Phi))
        R = np.diag(np.multiply(y,(1-y)))
        H = Phi.dot(R).dot(Phi.T)
        wStep = (np.linalg.inv(H)).dot(Phi).dot(y-tlabel)
        w = w - wStep
        stepSize = np.linalg.norm(wStep)
        print("Iter {0:3d}:      step {1:21.20f}".format(i,stepSize))
        if (stepSize < wToler):
            print("Within tolerance: quit")
            break
    return w
#-----------------------------------------------------------------------------------------------------------------------
# 1) Generating data
default_data_num = 200

# Class 0
theta0 = 0
m0 = np.zeros(dimen)
lamb0 = [2,1]
U0 = np.mat([[np.cos(theta0), np.sin(theta0)],[-np.sin(theta0), np.cos(theta0)]]).T

x0, C0 = gaussianFromEigen(m0, lamb0, U0, default_data_num)

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

x1A, C1A = gaussianFromEigen(mA, lambA, UA, default_data_num)
x1B, C1B = gaussianFromEigen(mB, lambB, UB, default_data_num)
mixGaussPi = np.random.uniform(0.0,1.0,default_data_num)

x1 = np.concatenate((x1A[mixGaussPi <= pi_A,:],x1B[mixGaussPi > pi_A,:]),axis=0)

print('C0 =\n',C0, '\n C1A =\n',C1A,'\n C1B =\n',C1B)

x = np.concatenate((x0,x1),axis=0) # data
t = np.concatenate((np.zeros(default_data_num),np.ones(default_data_num))) # label
#-----------------------------------------------------------------------------------------------------------------------
# 2)
# xRange = np.arange(np.min(x[:,0]),np.max(x[:,0]),0.05)
# yRange = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.05)
# xGrid, yGrid = np.meshgrid(xRange, yRange, sparse=False, indexing='xy')
# xGrid = np.reshape(xGrid, (xGrid.size,1))
# yGrid = np.reshape(yGrid, (yGrid.size,1))
# deciBoundX = np.column_stack((xGrid,yGrid))
# classPr = binaryMAP(deciBoundX)
# boundToler = 0.005;
# boundInd = ( np.abs(classPr[:,1]-classPr[:,0]) < boundToler)

classPr = binaryMAP(x)
mapPredict = ( (classPr[:,1]-classPr[:,0]) >= 0 ).astype(int)
incorrInd = np.squeeze(np.asarray((mapPredict.flatten() != t)))

# Visualize data and classification error (and decision boundary)
# plt.figure(figsize=(12, 9))
# # plt.scatter(xGrid[boundInd],yGrid[boundInd],s=0.5,c='g')
# plt.scatter(x0[:,0],x0[:,1],s=5,label='Class 0')
# plt.scatter(x1[:,0],x1[:,1],s=5,c='r',label='Class 1')
# plt.scatter(x[incorrInd,0],x[incorrInd,1],s=16,facecolors='none',edgecolors='k',label='Incorrect Classification')
# plt.title('MAP Classifier',fontsize=12)
# plt.xlabel('Dimension 0',fontsize=10)
# plt.ylabel('Dimension 1',fontsize=10)
# ax1 = plt.gca()
# ax1.set_aspect('equal', 'box')
# ax1.legend(loc='upper left')
#plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# 3)
incorrPr0 = np.sum(incorrInd[:default_data_num])/default_data_num
incorrPr1 = np.sum(incorrInd[default_data_num:])/default_data_num
print("MAP classifier:")
print('Conditional probability of incorrect classification of Class 0 = ',incorrPr0)
print('Conditional probability of incorrect classification of Class 1 = ',incorrPr1)

#-----------------------------------------------------------------------------------------------------------------------
# 4)
data_num = 200
x10,_ = gaussianFromEigen(m0, lamb0, U0, data_num)

x11A,_ = gaussianFromEigen(mA, lambA, UA, data_num)
x11B,_ = gaussianFromEigen(mB, lambB, UB, data_num)
mixGaussPi2 = np.random.uniform(0.0,1.0,data_num)
x11 = np.concatenate((x11A[mixGaussPi2 <= pi_A,:],x11B[mixGaussPi2 > pi_A,:]),axis=0)
x11 = x11[np.random.permutation(data_num),:] # Reshuffle the gaussian mixture data

xTrain = np.concatenate((x10,x11),axis=0) # data
tTrain = np.concatenate((np.zeros(data_num),np.ones(data_num))) # label

# trainInd = list(range(0,100)) + list(range(data_num,data_num+100))
# xTrain = x1x[trainInd,:]
# tTrain = t1x[trainInd]

#-----------------------------------------------------------------------------------------------------------------------
# 5)
#-----------------------------------------------------------------------------------------------------------------------
# 6)
#-----------------------------------------------------------------------------------------------------------------------
# 7)
Phi = nonKer10Feature(xTrain)
w = trainLRBC(Phi, tTrain)
lrPredict = predictLRBC(Phi,w)
incorrInd = np.squeeze(np.asarray((lrPredict.flatten() != tTrain)))

incorrPr0 = np.sum(incorrInd[:data_num])/data_num
incorrPr1 = np.sum(incorrInd[data_num:])/data_num
print('Training error of Class 0 = ',incorrPr0)
print('Training error of Class 1 = ',incorrPr1)

# Visualize data and classification error (and decision boundary)
# plt.figure(figsize=(12, 9))
# # plt.scatter(xGrid[boundInd],yGrid[boundInd],s=0.5,c='g')
# plt.scatter(xTrain[:data_num,0],xTrain[:data_num,1],s=5,label='Class 0')
# plt.scatter(xTrain[data_num:,0],xTrain[data_num:,1],s=5,c='r',label='Class 1')
# plt.scatter(xTrain[incorrInd,0],xTrain[incorrInd,1],s=16,facecolors='none',edgecolors='k',label='Incorrect Classification')
# plt.title('Training logistic regression Classifier',fontsize=12)
# plt.xlabel('Dimension 0',fontsize=10)
# plt.ylabel('Dimension 1',fontsize=10)
# ax1 = plt.gca()
# ax1.set_aspect('equal', 'box')
# ax1.legend(loc='upper left')
# plt.show()

lrPredict = predictLRBC(nonKer10Feature(x),w)
incorrInd = np.squeeze(np.asarray((lrPredict.flatten() != t)))
incorrPr0 = np.sum(incorrInd[:default_data_num])/default_data_num
incorrPr1 = np.sum(incorrInd[default_data_num:])/default_data_num
print("Logistic regression classifier:")
print('Prediction error of Class 0 = ',incorrPr0)
print('Prediction error of Class 1 = ',incorrPr1)

#-----------------------------------------------------------------------------------------------------------------------
print('End')