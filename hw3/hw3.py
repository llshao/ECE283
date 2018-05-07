# Homework 3: Unsupervised Learning, part 1 (Gaussian mixtures and EM algorithm; K-means and soft K-means
# Created on 05/06/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans

#-----------------------------------------------------------------------------------------------------------------------
# Configurations
dimen = 2 # data dimension

learnRate = 0.1
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
# Functions
def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    Cov = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        Cov = Cov +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, Cov, data_num)
    return ret_data, Cov

# Visualize data
def disp2DResult(data_in, one_hot, disp_now=1):
    label_num = one_hot.shape[1]
    for i in range(label_num):
        plt.scatter(data_in[one_hot[:,i],0],data_in[one_hot[:,i],1],s=5,label="Component {:2d}".format(i+1))
    plt.title('Data visulization',fontsize=12)
    plt.xlabel('Dimension 0',fontsize=10)
    plt.ylabel('Dimension 1',fontsize=10)
    plt_ax = plt.gca()
    plt_ax.set_aspect('equal', 'box')
    plt_ax.legend(loc='upper left',bbox_to_anchor=(-0.3, 1.2))
    if disp_now:
        plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Experiments with 2D data
# Generating data
data_num = 200

# Component 0:
theta0 = 0
pi0 = 1/2
m0 = np.zeros(dimen)
lamb0 = [2,1]
U0 = np.mat([[np.cos(theta0), np.sin(theta0)],[-np.sin(theta0), np.cos(theta0)]]).T

# Component 1:
theta1 = -3*(np.pi)/4
pi1 = 1/6
m1 = np.array([-2,1])
lamb1 = [2,1/4]
U1 = np.mat([[np.cos(theta1), np.sin(theta1)],[-np.sin(theta1), np.cos(theta1)]]).T

# Component 2:
theta2 = (np.pi)/4
pi2 = 1/3
m2 = np.array([3,2])
lamb2 = [3,1]
U2 = np.mat([[np.cos(theta2), np.sin(theta2)],[-np.sin(theta2), np.cos(theta2)]]).T

x0, C0 = gaussianFromEigen(m0, lamb0, U0, data_num)
x1, C1 = gaussianFromEigen(m1, lamb1, U1, data_num)
x2, C2 = gaussianFromEigen(m2, lamb2, U2, data_num)

mixGaussInd = np.random.choice(3,data_num,p=[pi0,pi1,pi2])
ind0 = (mixGaussInd==0)
ind1 = (mixGaussInd==1)
ind2 = (mixGaussInd==2)
x0Len = np.sum(ind0)
x1Len = np.sum(ind1)
x2Len = np.sum(ind2)
print("Ratio of gaussian mixture components = {:d}:{:d}:{:d}".format(x0Len,x1Len,x2Len))

x2D = np.concatenate((x0[ind0,:], x1[ind1,:], x2[ind2,:]),axis=0)
temp = np.concatenate((np.zeros(x0Len),np.ones(x1Len),2*np.ones(x2Len)))
z2D = np.column_stack((temp==0,temp==1,temp==2)) # One-hot encoding (.astype(int))

# disp2DResult(x2D, z2D)
#-----------------------------------------------------------------------------------------------------------------------
# K-means algorithm
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.05, hspace=0.3)
for k in range(2,6):
    kmean_h = KMeans(n_clusters=k, init='random', n_init=10, random_state=0).fit(x2D)
    clust_center = kmean_h.cluster_centers_
    curr_clust = (kmean_h.labels_ == 0)
    for c_i in range(1,k):
        curr_clust = np.column_stack((curr_clust, kmean_h.labels_ == c_i))

    plt.subplot(gs[k-2])
    disp2DResult(x2D, curr_clust,0)
    plt.title("{:d} Clusters".format(k))
    plt.plot(clust_center[:,0],clust_center[:,1],'kx')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
print('End')