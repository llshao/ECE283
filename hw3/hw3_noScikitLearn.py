# Homework 3: Unsupervised Learning, part 1 (Gaussian mixtures and EM algorithm; K-means and soft K-means
# Created on 05/12/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import contextlib
import numpy as np
import numpy.matlib as matlib
import numpy.linalg as nla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse

#-----------------------------------------------------------------------------------------------------------------------
# Configurations
np.random.seed(1)

#-----------------------------------------------------------------------------------------------------------------------
# General functions
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = nla.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    Cov = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        Cov = Cov +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, Cov, data_num)
    return ret_data, Cov

def getGaussianLikelihood(data_in, mean_v, Cov):
    GaussZ = np.power(2 * np.pi, (Cov.shape[0] * 0.5)) * np.power(nla.det(Cov), 0.5)
    GaussP = nla.inv(Cov)
    x_len = data_in.shape[0]
    likelihood = np.empty(x_len)
    for i in range(x_len):
        likelihood[i] = np.exp(-0.5 * np.mat(data_in[i,:] - mean_v) *GaussP *(np.mat(data_in[i,:] - mean_v).T)) /GaussZ
    return likelihood

def getEmpProbTable(comp_num, k, true_label, pred_label, margin_prob):
    joint_prob = np.zeros((comp_num,k)) # P(a,z)
    for c_i in range(k):
        ind = (pred_label == c_i)
        if c_i == 0:
            curr_clust = ind
        else:
            curr_clust = np.column_stack((curr_clust, ind))

        for a_i in range(comp_num):
            joint_prob[a_i,c_i] = np.mean(np.logical_and(true_label[:,a_i],ind))
    emp_prob = joint_prob/margin_prob # # empirical probabilities P(a|z) = P(a,z)/P(z)
    with printoptions(precision=3, suppress=True):
        print("Empirical probability table:\n",emp_prob)
    return curr_clust

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
# Class
class KMeans:
    def __init__(self, n_clusters, n_init=1, n_iter=100):
        self.clust_num = n_clusters
        self.n_init = n_init
        self.n_iter = n_iter

    def assignment_step(self, curr_center):
        MSE = np.empty((self.data_len, self.clust_num))
        for j in range(self.clust_num):
            MSE[:, j] = nla.norm(self.data_in - curr_center[j, :], axis=1)
        assign_label = np.argmin(MSE, axis=1)
        return assign_label,MSE

    def update_step(self, assign_label):
        curr_center = np.empty((self.clust_num,self.data_dimen))
        for j in range(self.clust_num):
            curr_center[j,:] = np.mean(self.data_in[assign_label == j,:], axis=0)
        return curr_center

    def fit(self, data_in):
        self.data_in = data_in
        self.data_len = data_in.shape[0]
        self.data_dimen = data_in.shape[1]

        data_min = np.min(data_in, axis=0)
        data_max = np.max(data_in, axis=0)

        all_center = []
        all_label = []
        all_MSE = np.empty((self.n_init))
        for i in range(self.n_init):
            invalid_init = 1
            while(invalid_init):
                curr_center = np.random.uniform(data_min,data_max,size=(self.clust_num,self.data_dimen))
                assign_label, _ = self.assignment_step(curr_center)
                label_num = len(np.unique(assign_label))
                if (label_num == self.clust_num):
                    invalid_init = 0

            for k in range(self.n_iter):
                curr_center = self.update_step(assign_label)
                assign_label,MSE = self.assignment_step(curr_center)
            all_center.append(curr_center)
            all_label.append(assign_label)
            all_MSE[i] = np.sum(MSE)
        slct_init_ind = np.argmin(all_MSE)
        self.cluster_centers_ = all_center[slct_init_ind]
        self.labels_ = all_label[slct_init_ind]
        return self

class GaussianMixture(KMeans):
    def __init__(self,n_components, n_init=1, n_iter=100):
        super().__init__(n_components, n_init, n_iter)

    def E_step(self,data_in):
        prob_kx = np.empty((self.clust_num,self.data_len))
        for j in range(self.clust_num):
            prob_kx[j,:] = getGaussianLikelihood(data_in, self.means_[j,:], self.covariances_[j,:,:])
        prob_kx = ( np.diag(self.weights_).dot(prob_kx) ) / ( self.weights_.dot(prob_kx) )
        return prob_kx

    def M_step(self,prob_kx):
        prob_sum = np.sum(prob_kx,axis=1)

        self.means_ = prob_kx.dot(self.data_in) / (prob_sum.reshape((self.clust_num,1)))

        for j in range(self.clust_num):
            self.covariances_[j, :, :] = (( np.diag(prob_kx[j,:]).dot(self.data_in - self.means_[j, :]).T)
                                          .dot(self.data_in - self.means_[j, :])) / prob_sum[j]

        self.weights_ = prob_sum / self.data_len

    def fit(self, data_in):
        super().fit(data_in)
        self.means_ = self.cluster_centers_

        self.covariances_ = np.empty((self.clust_num,self.data_dimen,self.data_dimen))
        prob_sum = np.empty((self.clust_num))
        for j in range(self.clust_num):
            prob_kx = (self.labels_ == j)
            prob_sum[j] = np.sum(prob_kx)
            self.covariances_[j,:,:] = ( ((self.data_in[prob_kx,:] - self.means_[j,:]).T)
                                 .dot(self.data_in[prob_kx,:] - self.means_[j,:]) ) / prob_sum[j]

        self.weights_ = prob_sum/self.data_len

        for i in range(self.n_init):
            prob_kx = self.E_step(self.data_in)
            self.M_step(prob_kx)
        return self

    def predict(self, data_in):
        prob_kx = self.E_step(data_in)
        self.labels = np.argmax(prob_kx,axis=0)
        return self.labels

#-----------------------------------------------------------------------------------------------------------------------
# Experiments with 2D data
dimen = 2 # data dimension

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
prob_z = np.array([x0Len, x1Len, x2Len]).reshape((3,1))/data_num # P(z)
print("Ratio of gaussian mixture components = {:d}:{:d}:{:d}".format(x0Len,x1Len,x2Len))
print("P(z[0]=1) = {:.3f}, P(z[1]=1) = {:.3f}, P(z[2]=1) = {:.3f}".format(prob_z[0,0],prob_z[1,0],prob_z[2,0]))

x2D = np.concatenate((x0[ind0,:], x1[ind1,:], x2[ind2,:]),axis=0)
temp = np.concatenate((np.zeros(x0Len),np.ones(x1Len),2*np.ones(x2Len)))
z2D = np.column_stack((temp==0,temp==1,temp==2)) # One-hot encoding (.astype(int))

plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
disp2DResult(x2D, z2D,0)
plt.legend(loc='upper left',bbox_to_anchor=(-0.3, 1.1))
w_factor = 0.1
draw_ellipse(m0, C0, alpha=pi0 * w_factor, color='k')
draw_ellipse(m1, C1, alpha=pi1 * w_factor, color='k')
draw_ellipse(m2, C2, alpha=pi2 * w_factor, color='k')

#-----------------------------------------------------------------------------------------------------------------------
# 1) K-means algorithm
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.05, hspace=0.3)

print("------------------------- K-means algorithm -------------------------")
clust_center = []
for k in range(2,6):
    print("k = ",k)
    kmean_h = KMeans(n_clusters=k, n_init=10).fit(x2D)
    clust_center.append(kmean_h.cluster_centers_)

    curr_clust = getEmpProbTable(3, k, z2D, kmean_h.labels_, prob_z)

    plt.subplot(gs[k-2])
    disp2DResult(x2D, curr_clust,0)
    plt.title("{:d} Clusters".format(k))
    plt.plot(clust_center[k-2][:,0],clust_center[k-2][:,1],'kx')

#-----------------------------------------------------------------------------------------------------------------------
# 2-3) Estimate the mean and covariance for a Gaussian mixture model
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.05, hspace=0.3)

print("------------------------- Gaussian mixture model -------------------------")
gauss_mean = []
for k in range(2,6):
    print("k = ", k)
    gmm_h = GaussianMixture(n_components=k, n_init=10).fit(x2D)
    gauss_mean.append(gmm_h.means_)
    labels = gmm_h.predict(x2D)

    curr_clust = getEmpProbTable(3, k, z2D, labels, prob_z)

    plt.subplot(gs[k-2])
    disp2DResult(x2D, curr_clust,0)
    plt.title("{:d} Components".format(k))
    plt.plot(gauss_mean[k-2][:,0],gauss_mean[k-2][:,1],'kx')

    w_factor = 0.05 / gmm_h.weights_.max()
    for pos, covar, w in zip(gmm_h.means_, gmm_h.covariances_, gmm_h.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor, color='k')

plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# 4) Generate seven "quasi-orthogonal" random vectors in d dimensions
d = 30 # increased dimension
Pu0 = 2/3 # P(u[i] = 0)
Pu1 = 1/6 # P(u[i] = +1)
Pu2 = 1/6 # P(u[i] = -1)
vect_pool_size = 7
not_ortho = 1
vect_num = 0

vect_pool = np.random.choice([0,1,-1], size=[7,d], p=[Pu0, Pu1, Pu2])

ortho_thresh = 1 # threshold can be adjusted
diag_ind = np.diag_indices(vect_pool_size)
iter_i = 0
while not_ortho:
    dot_prod = (vect_pool).dot(vect_pool.T)
    dot_prod[diag_ind] = 0
    corr_score = np.sum(np.absolute(dot_prod),axis=1)
    # print("iter ",iter_i,"corr_score = ",corr_score)
    if (np.max(corr_score) == 0) or (np.sum(corr_score) < ortho_thresh):
        not_ortho = 0
    else:
        iter_i += 1
        drop_ind = np.argmax(corr_score)
        vect_pool[drop_ind,:] = np.random.choice([0,1,-1], d, p=[Pu0, Pu1, Pu2])
print("[u1,...,u7] = \n",vect_pool.T)

#-----------------------------------------------------------------------------------------------------------------------
# 5-6) Generate 30-dimensional data samples for a Gaussian mixture distribution with 3 equiprobable components and run
#      the K-means algorithm with different values of K

sigma = 0.01 # noise level
data_num = 200 # data number to generate

Z1 = np.random.normal(0, 1, data_num)
Z2 = np.random.normal(0, 1, data_num)
N = np.random.normal(0, sigma, size=[data_num,d])

Comp1 = vect_pool[0,:] + np.outer(Z1,vect_pool[1,:]) + np.outer(Z2,vect_pool[2,:]) + N
Comp2 = 2*vect_pool[3,:] + (2**0.5)*np.outer(Z1,vect_pool[4,:]) + np.outer(Z2,vect_pool[5,:]) + N
Comp3 = (2**0.5)*vect_pool[5,:] + np.outer(Z1,(vect_pool[0,:]+vect_pool[1,:])) + \
        0.5*(2**0.5)*np.outer(Z2,vect_pool[4,:]) + N

# (E[X])
Comp_mean = np.zeros((3,d))
Comp_mean[0,:] = vect_pool[0,:]
Comp_mean[1,:] = 2*vect_pool[3,:]
Comp_mean[2,:] = (2**0.5)*vect_pool[5,:]

# (Var(X) = E[X^2] - E[X]^2 => Var(X1+X2) = Var(X1) + Var(X2) if X1 and X2 are independent)
Comp_var = np.zeros((3,d))
Comp_var[0,:] = np.square(vect_pool[1,:]) + np.square(vect_pool[2,:]) + (sigma**2)*np.ones(d)
Comp_var[1,:] = 2*np.square(vect_pool[4,:]) + np.square(vect_pool[5,:]) + (sigma**2)*np.ones(d)
Comp_var[2,:] = np.square(vect_pool[0,:]+vect_pool[1,:]) + 0.5*np.square(vect_pool[4,:]) + (sigma**2)*np.ones(d)
# Comp_eig_vect = np.sqrt(Comp_var)
# print("Component covariance matrix eigenvector (transposed): \n", Comp_eig_vect.T)

mixGaussInd = np.random.choice(3,data_num)
ind0 = (mixGaussInd==0)
ind1 = (mixGaussInd==1)
ind2 = (mixGaussInd==2)
x0Len = np.sum(ind0)
x1Len = np.sum(ind1)
x2Len = np.sum(ind2)
prob_z = np.array([x0Len, x1Len, x2Len]).reshape((3,1))/data_num # P(z)
print("Ratio of gaussian mixture components (30D) = {:d}:{:d}:{:d}".format(x0Len,x1Len,x2Len))
print("P(z[0]=1) = {:.3f}, P(z[1]=1) = {:.3f}, P(z[2]=1) = {:.3f}".format(prob_z[0,0],prob_z[1,0],prob_z[2,0]))
x30D = np.concatenate((Comp1[ind0,:], Comp2[ind1,:], Comp3[ind2,:]),axis=0)
temp = np.concatenate((np.zeros(x0Len),np.ones(x1Len),2*np.ones(x2Len)))
z30D = np.column_stack((temp==0,temp==1,temp==2)) # One-hot encoding (.astype(int))

print("------------------------- K-means of data 30D -------------------------")
clust_center = []
for k in range(2,6):
    print("k = ",k)
    kmean_h = KMeans(n_clusters=k, n_init=10).fit(x30D)
    clust_center.append(kmean_h.cluster_centers_)

    getEmpProbTable(3, k, z30D, kmean_h.labels_, prob_z)

#-----------------------------------------------------------------------------------------------------------------------
# 7) How the cluster centers found by K-means relate to the seven vectors
for i in range(4):
    clust_vect_corr = (Comp_mean/nla.norm(Comp_mean,axis=1).reshape((Comp_mean.shape[0],1))) \
        .dot((clust_center[i]/nla.norm(clust_center[i],axis=1).reshape((clust_center[i].shape[0],1))).T)
    with printoptions(precision=1, suppress=True):
        print("k={:2d}: Correlation between cluster mean and data model mean:\n".format(i+2),clust_vect_corr)

#-----------------------------------------------------------------------------------------------------------------------
# 8) Run EM algorithm with several different values of K.
print("------------------------- Gaussian mixture of data 30D -------------------------")
for k in range(3,6):
    print("k = ",k)
    # gmm_h = GaussianMixture(n_components=k, covariance_type='diag').fit(x30D)
    gmm_h = GaussianMixture(n_components=k, n_init=10).fit(x30D)
    labels = gmm_h.predict(x30D)
    getEmpProbTable(3, k, z30D, labels, prob_z)

    gauss_mean = gmm_h.means_
    clust_vect_corr = (Comp_mean / nla.norm(Comp_mean, axis=1).reshape((Comp_mean.shape[0], 1))) \
        .dot((gauss_mean / nla.norm(gauss_mean, axis=1).reshape((gauss_mean.shape[0], 1))).T)
    with printoptions(precision=1, suppress=True):
        print("Correlation between GMM component mean and data model mean:\n", clust_vect_corr)

    gauss_cov = np.empty((k,d))
    for j in range(k):
        gauss_cov[j,:] = np.diag(gmm_h.covariances_[j,:,:])

    clust_vect_corr = (Comp_var / nla.norm(Comp_var, axis=1).reshape((Comp_var.shape[0], 1))) \
        .dot((gauss_cov / nla.norm(gauss_cov, axis=1).reshape((gauss_cov.shape[0], 1))).T)
    with printoptions(precision=1, suppress=True):
        print("Correlation between GMM component covariance and data model covariance:\n", clust_vect_corr)
        print("Component covariance matrix (diagonal, transposed): \n", Comp_var.T)
        print("GMM component covariance matrix (Most match, re-ordered, transposed): \n",
              gauss_cov[np.argmax(clust_vect_corr, axis=1), :].T)

#-----------------------------------------------------------------------------------------------------------------------
pass