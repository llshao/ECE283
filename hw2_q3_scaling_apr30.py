# Homework 2

import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os.path
import math
#-----------------------------------------------------------------------------------------------------------------------
# Functions
def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    Cov = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        Cov = Cov +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, Cov, data_num)
    return ret_data, Cov
#-----------------------------------------------------------------------------------------------------------------------
# Define NN layers weight & bias
def creat_weightbais(num_input,num_classes,n_hidden_1,n_hidden_2):
	weights = {
		'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
	}
	#weights = {
	#	'h1': tf.Variable(tf.random_uniform([num_input, n_hidden_1])),
	#	'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2])),
	#	'out': tf.Variable(tf.random_uniform([n_hidden_2, num_classes]))
	#}
	
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([num_classes]))
	}
	return weights,biases

# Create model
def neural_net(x,weights,biases):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def Initilize_NN(num_input,num_classes,n_hidden_1,n_hidden_2,reg_const,learning_rate):
	#-----------------------------------------------------------------------------------------------------------------------
	# TensorFlow initialization
	X = tf.placeholder(tf.float32, [None, num_input])
	Y = tf.placeholder(tf.float32, [None, num_classes])
	# Construct lost & optimization model
	weights,biases=creat_weightbais(num_input,num_classes,n_hidden_1,n_hidden_2)
	logits = neural_net(X,weights,biases)
	output = tf.sigmoid(logits)
	# output = tf.nn.softmax(logits) ## for multi-classes
	# Define loss and optimizer
	# loss_op = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
	# softmax_cross_entropy_with_logits & sigmoid_cross_entropy_with_logits internally do the sigmoid and softmax to y
	# lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
	# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	# loss_op = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))   
	
	regularizer1 = reg_const*tf.reduce_mean(tf.nn.l2_loss(weights['h1'])+tf.nn.l2_loss(weights['h2']) +tf.nn.l2_loss(weights['out']))  # L2 regularization on weight of hidden layer 1
	loss_op += regularizer1 
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)
	# Evaluate model for multi-classes
	# correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
	# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	prediction = tf.cast(tf.greater(output, 0.5), tf.float32) # Threshold output to 0 and 1 to get predicted labels
	accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y), tf.float32)) # Compare prediction to true labels
	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
	return X,Y,train_op,loss_op,accuracy,init


def GenerateSample(dimen,default_date_num):
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

	#-----------------------------------------------------------------------------------------------------------------------
	# prepare training, validation and testing data set
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
	return xTrain,tTrain,xValid,tValid,xTest,tTest,trainLen
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Generating data
np.random.seed(0)
dimen = 2 # data dimension
default_data_num = 10000
xTrain0,tTrain0,xValid0,tValid0,xTest0,tTest0,trainLen0=GenerateSample(dimen,default_data_num)
#print(XMEAN,XSTD)
# NN Configurations
reg_const = 0.00 # regulation weight
learning_rate = 0.005
# Network parameters
n_hidden_1 = 64 # number of neurons in hidden layer 1
n_hidden_2 = 64 # number of neurons in hidden layer 2
num_input = dimen # input data dimension
num_classes = 1 # number of classes
# Training loop
N_epochs = 100 #training loop

save_path = '/home/cadlab/Mytest/'
name = os.path.join(save_path, 'hw2_q3_scaling.txt')
file = open(name, 'w')
HIDDEN1_LIST=[16]
HIDDEN2_LIST=[32]
LEARN_LIST	=[0.005]
REG_LIST	=[0.00]
BATCH_LIST	=[10]
DATA_NUM_LIST=[200]
NORM		= False
loss_all	= []
epoch_all   = []
file.write('n_hidden_1  n_hidden_2 learning_rate  reg_const batchSize data_num acc_valid acc_test\n')
for n_hidden_1 in HIDDEN1_LIST:
	for n_hidden_2 in HIDDEN2_LIST:
		for learning_rate in LEARN_LIST:
			for reg_const in REG_LIST:
				for batchNum in BATCH_LIST:
					for data_num in DATA_NUM_LIST:
						trainLen=int((trainLen0*data_num/default_data_num))
						testLen =int((trainLen0/7*2*data_num/default_data_num))
						validLen=int((trainLen0/7*data_num/default_data_num))
						print('trainLeb {:d},testLen{:d}, validLen {:d}'.format(trainLen,testLen,validLen))
						xTrain	= xTrain0[0:trainLen,:]
						tTrain	= tTrain0[0:trainLen]
						xValid	= xValid0[0:validLen,:]
						tValid	= tValid0[0:validLen]
						xTest	= xTest0[0:testLen,:]
						tTest	= tTest0[0:testLen]
						xAll=np.concatenate((xTrain,xTest,xValid),axis=0)
						XMEAN0 = np.mean(xAll,axis=0,dtype=np.float32)
						XSTD0  = np.std(xAll,axis=0,dtype=np.float32)
						XMEAN = np.mean(xTrain,axis=0,dtype=np.float32)
						XSTD  = np.std(xTrain,axis=0,dtype=np.float32)
						if NORM==True:
							xTrain -= XMEAN0#np.mean(xTrain,axis=0,dtype=np.float32)
							xTrain /= XSTD0#np.std(xTrain,axis=0,dtype=np.float32)
							print(np.mean(xTrain,axis=0),np.std(xTrain,axis=0))
							xValid -= XMEAN0#np.mean(xValid,axis=0,dtype=np.float32)
							xValid /= XSTD0#np.std(xValid,axis=0,dtype=np.float32)
							xTest  -= XMEAN0#np.mean(xTest,axis=0,dtype=np.float32)
							xTest  /= XSTD0#np.std(xTest,axis=0,dtype=np.float32)
						batchSize = int(np.floor(trainLen/batchNum))
						print('\n n_hidden_1 = {:d}, n_hidden_2={:d}, learning_rate = {:03f}, reg_const = {:03f}, batchSize = {:d}, data_num={:d}'.format(n_hidden_1,n_hidden_2, learning_rate, reg_const, batchNum,data_num))
						file.write('{:d}, {:d}, {:03f}, {:03f}, {:d}, {:d},'.format(n_hidden_1,n_hidden_2, learning_rate, reg_const, batchNum, data_num))
					#-----------------------------------------------------------------------------------------------------------------------
						# 1) implement a fully connected neural network
						# (a) 1 hidden layer
						# (b) 2 hidden layer
						tf.reset_default_graph()
						sess = tf.InteractiveSession()
						X,Y,train_op,loss_op,accuracy,init=Initilize_NN(num_input,num_classes,n_hidden_1,n_hidden_2,reg_const,learning_rate)				#-----------------------------------------------------------------------------------------------------------------------
						# start training
						with tf.Session() as sess:
							sess.run(init)
							print('\nTraining\n')
							for epoch in range(N_epochs):
								#print("Epochs ",epoch)
								randInd = np.random.permutation(trainLen)
								xTrain = xTrain[randInd, :]
								tTrain = tTrain[randInd]
								for step in range(batchNum):
									currInd = range(step * batchSize, (step+1) * batchSize)
									batch_x = xTrain[currInd, :]
									batch_y = tTrain[currInd]
									batch_y = batch_y.reshape(batchSize,num_classes)
									#batch_y = np.column_stack((1-batch_y,batch_y)) # one-hot encoding
									# Run optimization op (backprop)
									sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

									# Calculate batch loss and accuracy
								#if epoch % 10 == 0:
									#loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
									#loss_all.append(loss)
									#epoch_all.append(epoch)
									#print("Epochs {0:3d}: Minibatch loss = {1:5.4f} , Training accuracy = {2:4.3f}".format(epoch,loss,acc))
									#tValid = tValid.reshape((tValid.size,num_classes))
									#print("Validate accuracy = {:4.3f}".format(sess.run(accuracy, feed_dict={X: xValid, Y: tValid})))
							#print("Optimization finished!")
							# Calculate accuracy for test data
							# tTest = np.column_stack((1 - tTest, tTest))  # one-hot encoding
							tTest = tTest.reshape(tTest.size,num_classes)
							tValid = tValid.reshape((tValid.size,num_classes))
							acc_valid=sess.run(accuracy, feed_dict={X: xValid, Y: tValid})
							acc_test=sess.run(accuracy, feed_dict={X: xTest, Y: tTest})
							print('\n Valid Accuracy {:04f} Test Accuracy {:04f}\n'.format(acc_valid,acc_test))
							file.write('{:04f}, {:04f}\n'.format(acc_valid,acc_test))
							sess.close()
file.close()
#plt.figure()
#plt.plot(epoch_all,loss_all)
#plt.show()

					#print("\nTest accuracy = {:4.3f}".format(sess.run(accuracy, feed_dict={X: xTest, Y: tTest})))
