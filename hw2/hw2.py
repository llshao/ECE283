# Homework 2: Classification using neural networks
# Created on 04/23/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import tensorflow as tf

#-----------------------------------------------------------------------------------------------------------------------
# Configurations
dimen = 2 # data dimension

learnRate = 0.005
stepNum = 10

suffleNum = 5

# Network parameters
numHL1 = 64 # number of neurons in hidden layer 1
numHL2 = 64 # number of neurons in hidden layer 2

l2Factor = 1 # significance of L2 regularization

inputDim = dimen # input data dimension
numClass = 2 # number of classes

np.random.seed(0)
#-----------------------------------------------------------------------------------------------------------------------
# TensorFlow initialization
X = tf.placeholder(tf.float32, [None, inputDim])
Y = tf.placeholder(tf.float32, [None, numClass])

# Store layers weight and bias
# tf.Variable will be updated
weights = {
    'h1': tf.Variable(tf.random_normal([inputDim, numHL1])),
    'h2': tf.Variable(tf.random_normal([numHL1, numHL2])),
    'out': tf.Variable(tf.random_normal([numHL2, numClass]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([numHL1])),
    'b2': tf.Variable(tf.random_normal([numHL2])),
    'out': tf.Variable(tf.random_normal([numClass]))
}

# Create neural network model
def neural_net(x):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.relu(layer2)

    layerOut = tf.matmul(layer2, weights['out']) + biases['out']
    # layerOut = tf.sigmoid(layerOut)

    return layerOut

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
# lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# lossOp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
lossOp = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))

# # tf.losses.sigmoid_cross_entropy
# L_BCS(y\hat,y) = -(ylogy\hat+(1-y)log(1-y\hat))
# L_sigmoid_BCS(y\hat,y) = -(ylog\sigmoid(y\hat)+(1-y)log(1-\sigmoid(y\hat)))???

# regularizer1 = tf.nn.l2_loss(weights['h1']) # L2 regularization on weight of hidden layer 1
# regularizer2 = tf.nn.l2_loss(weights['h2']) # L2 regularization on weight of hidden layer 2
# lossOp += tf.reduce_mean(l2Factor*regularizer1 + l2Factor*regularizer2)

optimizer = tf.train.AdamOptimizer(learning_rate=learnRate)
trainOp = optimizer.minimize(lossOp)

# Evaluate model
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# Initialize the variables (assign their default value)
init = tf.global_variables_initializer()

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
default_data_num = 800

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
trainLen = xTrain.shape[0]
randInd = np.random.permutation(trainLen)
xTrain = xTrain[randInd,:]
tTrain = tTrain[randInd]

batchSize = int(np.floor(trainLen/stepNum))

#-----------------------------------------------------------------------------------------------------------------------
# 1) implement a fully connected neural network
# (a) 1 hidden layer

# (b) 2 hidden layer
with tf.Session() as sess:
    sess.run(init)

    for s_i in range(suffleNum):
        print("Suffle ",s_i)
        randInd = np.random.permutation(trainLen)
        xTrain = xTrain[randInd, :]
        tTrain = tTrain[randInd]
        for step in range(stepNum):
            currInd = range(step * batchSize, (step+1) * batchSize)
            batch_x = xTrain[currInd, :]
            batch_y = tTrain[currInd]
            batch_y = np.column_stack((1-batch_y,batch_y)) # one-hot encoding
            # Run optimization op (backprop)
            sess.run(trainOp, feed_dict={X: batch_x, Y: batch_y})

            # Calculate batch loss and accuracy
            loss, acc = sess.run([lossOp, accuracy], feed_dict={X: batch_x, Y: batch_y})

            print("Step {0:3d}: Minibatch loss = {1:5.4f} , Training accuracy = {2:4.3f}".format(step,loss,acc))

        print("Optimization finished!")

        # Calculate accuracy for test data
        testY = np.column_stack((1 - tTest, tTest))  # one-hot encoding
        print("Test accuracy = {:4.3f}".format(sess.run(accuracy, feed_dict={X: xTest, Y: testY})))

#-----------------------------------------------------------------------------------------------------------------------
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