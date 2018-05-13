# Homework 3: Unsupervised Learning, part 1 (Gaussian mixtures and EM algorithm; K-means and soft K-means
# Created on 05/06/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import contextlib
import numpy as np
import numpy.matlib as matlib
import numpy.linalg as nla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#----
