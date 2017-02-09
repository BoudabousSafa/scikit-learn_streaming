from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import KMeans
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

    def __init__(self, time_window=1, timestamp=-1, initialized=False, max_num_kernels=100, nb_initial_points , kernel_radifactor=2,
                 kernels=None, m=None):
        self.time_window = time_window  # Range of the window
        self.timestamp = timestamp
        self.initialized = initialized
        self.buffer =
        self.max_num_kernels = max_num_kernels  # maxNumKernels
        self.kernel_radifactor = kernel_radifactor  #
        self.kernels = kernels
        self.bufferSize = maxNumKernels
        self.m = maxNumKernels
        self.nb_initial_points = nb_initial_points

    def fit(self, X, Y=None):
    # use kmeans to generate the q microclusters
    # initialisation
        X= check_array(X, accept_sparse='csr')
        nb_initial_points = X.shape[0]
        if nb_initial_points > self.init_points_option:
            kmeans = KMeans(n_clusters=self.max_num_kernels, random_state=1)
            m_cluster_labels = kmeans.fit_predict(X, Y)
            X = np.column_stack((m_cluster_labels,X))
            initial_clusters = [ X[X[:,0] == str(l)][:,1:] for l in set(m_cluster_labels) if l != -1]
            [ self.create_micro_cluster(cluster) for cluster in initial_clusters ]

    def partial_fit(self, x, y):
        # kernel is same as microcluster !
        # 1. Determine Closest kernel
        closestKernel = None
        minDistance = sys.float_info.max
        for i in range(self.kernels):
            dist = get_distance(x, self.kernels[i])
            if (dist < minDistance):
                closestKernel = self.kernels[i]
                minDistancce = dist

        # 2. Check whether instance fits into closestKernels
        radius = 0.
        if (closestKernel.get_weight() == 1):
            radius = sys.float_info.max
            center = closestKernel.get_center()
            for i in range(self.kernels):
                if self.kernels[i] == closestKernel:
                    continue
                distance = get_distance(self.kernels[i], center)
                radius = min(distance, radius)
        else:
            radius = closestKernel.get_radius()
        
        if minDistance < radius:
            closestKernel.insert(x, timestamp)
        else: 
            # 3. no fit, free space to insert new kernel
            threshold = self.timestamp - self.timeWindow

            # 3.1 remove kernels
            for i in range(self.kernels):
                if (self.kernels[i].get_relevancdStamp() < threshold):
                    self.kernels[i] =  CluMicroCluster(x, self.timestamp)  #

            # 3.2 merge two kernels
            closestA = 0
            closestB = 0
            minDistance = sys.float_info.max
            for i in range(self.kernels):
                centerA = self.kernels[i].get_center()
                for j:
                    dist = get_distance(centerA, self.kernels[j].get_center())
                    if dist < minDistance:
                        minDistance = dist
                        closestA = i
                        closestB = j

            assert (closestA != closestB)
            self.kernels[closestA].insert(self.kernels[closestB])
            self.kernels[closestB] = CluMicroCluster(x, self.timestamp)  #

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
