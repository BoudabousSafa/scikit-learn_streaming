from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.neighbors import NearestNeighbors
import math as math
import numpy as np


class CluStream(BaseEstimator, ClusterMixin):
    """Implementation of DenStream
       Read more in the :
       Parameters
       ----------
       Examples
       --------
       * >>> X =
       * >>> y =
       * >>> from sklearn.xxx import DenStream
       * >>> ds = DenStream(params)
       * >>> ds.fit(X, y) # doctest: +ELLIPSIS
       DenStream(...)
       * >>> print(ds.predict())
       See also
       --------
       Notes
       -----
       """

    def __init__(self, timeWindow=1, timestamp=-1, initialized=False, maxNumKernels=100, kernelRadiFactor=2, kernels=None,m=None):
        self.timeWindow = timeWindow #Range of the window
        self.timestamp = timestamp
        self.initialized = initialized
        self.buffer =
        
        self.maxNumKernels = maxNumKernels # maxNumKernels
        self.kernelRadiFactor = kernelRadiFactor #
        self.kernels = kernels
        
        self.bufferSize = maxNumKernels
        self.m = maxNumKernels
    
    def fit(self, X, Y=None):
        # use kmeans to generate the q microclusters
        
    def partial_fit(self, x, y):
        # kernel is same as microcluster ! 
        
        # 1. Determine Closest kernel
        closestKernel = None
        minDistance =  sys.float_info.max
        for i in range(self.kernels):
            dist = calculate_distance(x, self.kernels[i])
            if(dist < minDistance):
                closestKernel = self.kernels[i]
                minDistancce = dist

        # 2. Check whether instance fits into closestKernels
        
        #3. no fit, free space to insert new kernel
        
        # 3.1 remove kernels
        
        # 3.2 merge two kernels 
        
        

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X :
        Returns
        -------
        y :
        """
        
        knn = nearestNeighbour()
        result = knn.fit_predict(cluster_centers, None)
        return result
