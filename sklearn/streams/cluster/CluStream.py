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

    def __init__(self, timeWindow=1, timestamp=-1, initialized=False, maxNumKernels=100, kernelRadiFactor=2, kernels=None,m=None, n_cluster = 3):
        self.timeWindow = timeWindow #Range of the window
        self.timestamp = timestamp
        self.initialized = initialized
        self.buffer =
        
        self.maxNumKernels = maxNumKernels # maxNumKernels
        self.kernelRadiFactor = kernelRadiFactor #
        self.kernels = kernels
        
        self.bufferSize = maxNumKernels
        self.m = maxNumKernels
        
        self.n_cluster = n_cluster
        self.micro_clusters = None
    
    def fit(self, X, Y=None):
        # use kmeans to generate the q microclusters
        
        k_means = KMeans(init='k-means++', n_clusters=self.n_cluster, n_init=10)
        k_means.fit(X)
        for k in range(k_means.n_clusters):
            my_members = k_means.labels_ == k
            for i, x in enumerate(X):
                if(my_members[i] == True):
                    timeStamp += 1
                    micro_cluster = MicroCluster()
                    micro_cluster.insert(self, x, timestamp)
        self.micro_clusters.append(micro_cluster)        
        
    def partial_fit(self, x, y):
        # kernel is same as microcluster ! 
        
        # 1. Determine Closest kernel
        closestKernel = None
        minDistance =  sys.float_info.max
        for i in range(self.kernels):
            dist = get_distance(x, self.kernels[i])
            if(dist < minDistance):
                closestKernel = self.kernels[i] 
                minDistancce = dist

        # 2. Check whether instance fits into closestKernels
        radius = 0.
        if(closestKernel.get_weight() == 1):
            radius = sys.float_info.max
            center = closestKernel.get_center()
            for i in range(self.kernels):
                if self.kernels[i] == closestKernel:
                    continue
                distance = get_distance(self.kernels[i], center)
                radius = min(distance, radius)
        else: 
            radius = closestKernel.get_radius()
        if minDistance < radius :
            closestKernel.insert(x, timestamp)
            
                    
        #3. no fit, free space to insert new kernel
        threshold = timestamp - timeWindow 
        
        # 3.1 remove kernels
        for i in range(self.kernels):
            if(self.kernels[i].get_relevancdStamp() < threshold):
                self.kernels[i] = None #
                
        # 3.2 merge two kernels 
        closestA = 0
        closestB = 0
        minDistance = sys.float_info.max
        for i in range(self.kernels):
            centerA = self.kernels[i].get_center()
            for j :
                dist = get_distance(centerA, self.kernels[j].get_center())
                if dist < minDistance:
                    minDistance = dist
                    closestA = i
                    closestB = j
        
        assert(closestA ! = closestB)
        self.kernels[closestA].insert(self.kernels[closestB])
        self.kernels[closestB] = None #
        
        

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X :
        Returns
        -------
        y :
        """
        
        kmeans = kmeans()
        result = kmeans.fit_predict(cluster_centers, None)
        return result
