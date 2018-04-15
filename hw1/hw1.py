# Homework 1: Classification using logistic regression
# Created on 04/08/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import scipy.stats as scist
import scipy.io as sio

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

# # Polynomial coefficents of Gaussian random variable
# def getGaussCoeff(x2, m, Cov, beta=1):
#     logZ = np.log((2 * np.pi) * np.power(np.linalg.det(Cov), 0.5))
#     P = np.linalg.inv(Cov)
#     Pminor = P[1,0]+P[0,1]
#     polyCoeff = np.array([
#         -0.5*P[0,0],
#         -0.5*( Pminor*x2 -2*m[0]*P[0,0] -m[1]*Pminor ),
#         -0.5 * (P[0, 0] * (m[0] ** 2) - Pminor * (x2 - m[1]) * m[0] + P[1, 1] * ((x2 - m[1]) ** 2)) - logZ + np.log(
#             beta)
#     ])
#     return polyCoeff

# Construct 10-dimentional feature vector Phi
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

# Generate Gaussian-kernel feature K
def GaussKerFeature(data_in, l=1):
    alpha = 0.5/(l**2)
    dataLen = data_in.shape[0]
    K = np.zeros((dataLen,dataLen))
    for i in range(dataLen):
        for j in range(dataLen):
            x_diff = data_in[i,:] - data_in[j,:]
            K[i,j] = np.exp(- alpha * (x_diff.dot(x_diff)))
    return K

# logistic sigmoid functio
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

#  Predict using a (non-kernelized) logistic regression binary classifier
def predictLRBC(Phi,w):
    y = sigmoid(w.dot(Phi))
    return (y > 0.5).astype(int)

#  Training a logistic regression binary classifier
def trainLRBC(Phi, tlabel, regular = None, maxIter = 100, wToler = 1e-15, wInit = None):
    featureN = Phi.shape[0] # dimension of features (must be rows of Phi)

    if (wInit == 'zeros'):
        w = np.zeros(featureN)
    elif (wInit == 'random'):
        w = np.random.uniform(0.0,1.0,featureN)-0.5
    else:
        if wInit is not None:
            w = wInit
        else:
            print("w must be initialized")
            return None

    if (regular == 'L2'):
        print("L2-regularization applied")
        for i in range(maxIter):
            y = sigmoid(w.dot(Phi))
            R = np.diag(np.multiply(y,(1-y)))
            H = Phi.dot(R).dot(Phi.T) + 0.5*np.eye(featureN)
            wStep = (np.linalg.inv(H)).dot(Phi.dot(y-tlabel) + 0.5*w)
            w = w - wStep
            stepSize = np.linalg.norm(wStep)
            print("Iter {0:3d}:      step {1:21.20f}".format(i,stepSize))
            if (stepSize < wToler):
                print("Within tolerance: quit\n")
                break
    else:
        for i in range(maxIter):
            y = sigmoid(w.dot(Phi))
            R = np.diag(np.multiply(y,(1-y)))
            H = Phi.dot(R).dot(Phi.T)
            wStep = (np.linalg.inv(H)).dot(Phi.dot(y-tlabel))
            w = w - wStep
            stepSize = np.linalg.norm(wStep)
            print("Iter {0:3d}:      step {1:21.20f}".format(i,stepSize))
            if (stepSize < wToler):
                print("Within tolerance: quit\n")
                break
    return w

# Evaluate binary classification result
def evalBCResult(pLabel, tLabel):
    incorrInd = np.squeeze(np.asarray((pLabel.flatten() != tLabel)))
    class0Ind = (tLabel == 0)
    class1Ind = (tLabel == 1)
    incorrPr0 = np.sum(incorrInd[class0Ind])/(class0Ind.shape[0])
    incorrPr1 = np.sum(incorrInd[class1Ind])/(class1Ind.shape[0])
    print("Class 0 - error = {0:4.1f}%\nClass 1 - error = {1:4.1f}%\n".format(100*incorrPr0,100*incorrPr1))
    return incorrInd

# Visualize data and classification error (and decision boundary)
def dispBCResult(x, class0Len, incorrInd, titleStr = None, plt_ax = None):
    if (plt_ax == None):
        plt_ax = plt.gca()
    plt.scatter(x[:class0Len,0],x[:class0Len,1],s=5,label='Class 0')
    plt.scatter(x[class0Len:,0],x[class0Len:,1],s=5,c='r',label='Class 1')
    plt.scatter(x[incorrInd,0],x[incorrInd,1],s=16,facecolors='none',edgecolors='k',label='Incorrect Classification')
    if (titleStr):
        plt.title(titleStr,fontsize=12)
    plt.xlabel('Dimension 0',fontsize=10)
    plt.ylabel('Dimension 1',fontsize=10)
    plt_ax.set_aspect('equal', 'box')
    plt_ax.legend(loc='upper left',bbox_to_anchor=(0.4, 1.4))

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

print('C0 =\n',C0, '\n C1A =\n',C1A,'\n C1B =\n',C1B)

x = np.concatenate((x0,x1),axis=0) # data
t = np.concatenate((np.zeros(default_data_num),np.ones(default_data_num))) # label
#-----------------------------------------------------------------------------------------------------------------------
# 2)
classPr = binaryMAP(x)
mapPredict = ( (classPr[:,1]-classPr[:,0]) >= 0 ).astype(int)
incorrInd = np.squeeze(np.asarray((mapPredict.flatten() != t)))

#-----------------------------------------------------------------------------------------------------------------------
# Decision boundary (incorrect)
# x2 = 1.25
# f0 = getGaussCoeff(x2, m0, C0)
# f1A = getGaussCoeff(x2, mA, C1A, pi_A)
# f1B = getGaussCoeff(x2, mB, C1B, pi_B)
# fbound = f1A + f1B - f0

# print('Test2 = ',f1A[2])
# f1A[2] -= np.log(0.11)
# print('Polynomial = ',f1A)
# fbound = f1A

# xRes = np.roots(fbound)
# print('roots: ',xRes)
# classPr = binaryMAP([xRes[0],x2])
# Pr_t0 = getGaussianLikelihood([xRes[0],x2], mA, C1A)
# print('Probability = ',Pr_t0*pi_A)
#-----------------------------------------------------------------------------------------------------------------------
# 3)
print("MAP classifier:")
incorrInd = evalBCResult(mapPredict, t)

# Visualize data and classification error
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x660+20+20")

ax0=plt.subplot2grid((1, 3), (0, 0), rowspan=1, colspan=1)
dispBCResult(x, default_data_num, incorrInd, plt_ax = ax0, titleStr = 'MAP Classifier')

# xRange = np.arange(np.min(x[:,0]),np.max(x[:,0]),0.05)
# yRange = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.05)
# xGrid, yGrid = np.meshgrid(xRange, yRange, sparse=False, indexing='xy')
# xGrid = np.reshape(xGrid, (xGrid.size,1))
# yGrid = np.reshape(yGrid, (yGrid.size,1))
# deciBoundX = np.column_stack((xGrid,yGrid))
# classPr = binaryMAP(deciBoundX)
# # boundToler = 0.005;
# # boundInd = ( np.abs(classPr[:,1]-classPr[:,0]) < boundToler)
# boundInd = ((classPr[:,1]-classPr[:,0]) >= 0)
# plt.scatter(xGrid[boundInd],yGrid[boundInd],s=0.01,c='g')
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

# sio.savemat('trainData',dict([('xTrain', xTrain), ('tTrain', tTrain)]))

print("Gaussian kernel logistic regression classifier:")
# Generate Gaussian-kernel feature
l = 0.1 # kernel radius

batchSize = 200
batchNum = int(data_num/batchSize)

dataLen = tTrain.shape[0]
lrPredict = np.zeros(dataLen)
for i in range(batchNum):
    print("Batch {0:2d}".format(i))
    batchInd = (i*batchSize) + np.arange(batchSize)
    batchInd = np.concatenate((batchInd,int(0.5*dataLen) +batchInd))
    K = GaussKerFeature(xTrain[batchInd,:], l)

    # Training LRBC (Gaussian kernel)
    if (i==0):
        a = trainLRBC(K, tTrain[batchInd], regular = 'L2', wInit = 'zeros', maxIter = 20)
    else:
        a = trainLRBC(K, tTrain[batchInd], regular = 'L2', wInit = a, maxIter=20)
    # Evaluate training
    lrPredict[batchInd] = predictLRBC(K,a)
    evalBCResult(lrPredict[batchInd], tTrain[batchInd])

print("Training completed")
incorrInd = evalBCResult(lrPredict, tTrain)

#-----------------------------------------------------------------------------------------------------------------------
# 5) and 6)
# Predict using LRBC
print("data x and label t:")

batchNum = int(default_data_num/batchSize)

lrPredict = np.zeros(t.shape[0])

for i in range(batchNum):
    print("Batch {0:2d}".format(i))
    batchInd = (i*batchSize*2) + np.arange(batchSize*2)
    lrPredict[batchInd] = predictLRBC(GaussKerFeature(x[batchInd,:], l), a)

incorrInd = evalBCResult(lrPredict, t)

# Visualize data and classification error (and decision boundary)
ax1=plt.subplot2grid((1, 3), (0, 1), rowspan=1, colspan=1)
dispBCResult(x, default_data_num, incorrInd, plt_ax = ax1, titleStr = 'Gaussian kernel LR Classifier')

xRange = np.linspace(np.min(x[:,0]),np.max(x[:,0]),40)
yRange = np.linspace(np.min(x[:,1]),np.max(x[:,1]),40)
xGrid, yGrid = np.meshgrid(xRange, yRange, sparse=False, indexing='xy')
xGrid = np.reshape(xGrid, (xGrid.size,1))
yGrid = np.reshape(yGrid, (yGrid.size,1))
deciBoundX = np.column_stack((xGrid,yGrid))

dataLen = deciBoundX.shape[0]
classPr = np.zeros(dataLen)
batchNum = int(dataLen/(batchSize*2))
for i in range(batchNum):
    batchInd = (i*batchSize*2) + np.arange(batchSize*2)
    # classPr[batchInd] = sigmoid(a.dot(GaussKerFeature(deciBoundX[batchInd,:], l)))
    classPr[batchInd] = predictLRBC(GaussKerFeature(deciBoundX[batchInd, :], l), a)
boundInd = (classPr == 0)
# boundInd = (classPr > 0.5)
plt.scatter(xGrid[boundInd],yGrid[boundInd],s=0.1,c='g')
#-----------------------------------------------------------------------------------------------------------------------
# 7)
# Generate (non-kernelized) feature
Phi = nonKer10Feature(xTrain)

# Training LRBC (non-kernelized)
w = trainLRBC(Phi, tTrain, wInit = 'zeros')

print("(non-kernelized) Logistic regression classifier:")
# Predict using LRBC (non-kernelized)
lrPredict = predictLRBC(Phi,w)
incorrInd = evalBCResult(lrPredict, tTrain)

# dispBCResult(xTrain, data_num, incorrInd, plt_ax = ax11,
#              titleStr = '(Non-kernelized) Logistic regression Classifier')

print("data x and label t:")
lrPredict = predictLRBC(nonKer10Feature(x),w)
incorrInd = evalBCResult(lrPredict, t)

# Visualize data and classification error (and decision boundary)
ax2=plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)
dispBCResult(x, default_data_num, incorrInd, plt_ax = ax2, titleStr = '(Non-kernelized) LR Classifier')

plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1)
plt.show()