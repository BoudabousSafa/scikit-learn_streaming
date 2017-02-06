from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import math as math
import numpy as np


class DenStream(BaseEstimator, ClusterMixin):
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

    def __init__(self, h=1000, min_points=10, init_points_option=1000, weight_threshold=0.01, lembda=0.25,
                 epsilon=0.01, mu=1, beta=0.001, p_micro_cluster=None, o_micro_cluster=None, init_buffer=None,
                 timestamp=0, current_timestamp=0):
        self.h = h  # horizon : Range of the window.
        self.min_points = min_points  # Minimal number of points cluster has to contain.
        self.init_points_option = init_points_option  # Number of points to use for initialization."
        self.weight_threshold = weight_threshold
        self.lembda = lembda  # lambda is a defined structure in python, i put 'e' instead of 'a'
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.p_micro_cluster = p_micro_cluster
        self.o_micro_cluster = o_micro_cluster
        self.init_buffer = init_buffer
        self.timestamp = timestamp
        self.current_timestamp = current_timestamp
        self.tp = math.round(1 / self.lembda * math.log((self.beta * self.mu) / (self.beta * self.mu - 1))) + 1

    def fit(self, X, Y=None):
        # initialisation
        # offline dbscan
        X= check_array(X, accept_sparse='csr')
        nb_initial_points = X.shape[0]
        if nb_initial_points > self.init_points_option:
            dbscan = DBSCAN(eps= self.epsilon, min_samples = self.min_points)
            m_cluster_labels = dbscan.fit_predict(X, Y)
            X = np.column_stack((m_cluster_labels,X))
            initial_clusters = [ X[X[:,0] == str(l)][:,1:] for l in set(m_cluster_labels) if l != -1]
            [ self.create_micro_cluster(cluster) for cluster in initial_clusters ]

    def create_micro_cluster(self, cluster):
        linear_sum = np.zeros(cluster.shape[1])
        squared_sum = np.zeros(cluster.shape[1])
        new_m_cluster = self.MicroCluster(nb_points=0, linear_sum= linear_sum, squared_sum= squared_sum, update_timestamp=0, lembda=self.lembda)
        [new_m_cluster.insert(point, self.current_timestamp) for point in cluster]
        self.p_micro_cluster.append(new_m_cluster)

    def merge_point_to_p_cluster(self,x,clusters):
        cluster_centers = list(map((lambda i: i.getCenter()), clusters))
        # Apply scikit learn nearest neignbor function over p_micro_cluster_centers in order to get the nearest potential micro-cluster
        nearest_micro_cluster_center = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cluster_centers)
        for i in range(self.p_micro_cluster):
            if self.p_micro_cluster[i].getCenter() == nearest_micro_cluster_center:
                cluster_index = i
                break
        # self.candidate_cluster = filter(lambda cp: cp.getCenter()== self.nearest_p_micro_cluster, self.p_micro_cluster)
        e_cluster = self.p_micro_cluster[cluster_index]
        e_cluster.insert(x)
        if e_cluster.compute_radius(self.current_timestamp) <= self.epsilon:
            self.p_micro_cluster[cluster_index].insert(x)
            return True
        else:
            return False

    def merge_point_to_o_cluster(self,x,clusters):
        cluster_centers = list(map((lambda i: i.getCenter()), clusters))
        # Apply scikit learn nearest neignbor function over p_micro_cluster_centers in order to get the nearest potential micro-cluster
        nearest_micro_cluster_center = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cluster_centers)
        for i in range(self.p_micro_cluster):
            if self.p_micro_cluster[i].getCenter() == nearest_micro_cluster_center:
                cluster_index = i
                break
        # self.candidate_cluster = filter(lambda cp: cp.getCenter()== self.nearest_p_micro_cluster, self.p_micro_cluster)
        e_cluster = self.p_micro_cluster[cluster_index]
        e_cluster.insert(x)
        if e_cluster.compute_radius(self.current_timestamp) <= self.epsilon:
            self.o_micro_cluster[cluster_index].insert(x,self.current_timestamp)
            if(self.o_micro_cluster[cluster_index].getWeight(self.current_timestamp)> self.beta*self.mu):
                self.p_micro_cluster.append(self.o_micro_cluster[cluster_index])
                del self.o_micro_cluster[cluster_index]
            return True
        else:
            return False

    def partial_fit(self, x, y):
        self.timestamp +=1
        self.current_timestamp = self.timestamp
        # ajout d'instance
        merged = self.merge_point_to_p_cluster(x,self.p_micro_cluster)
        if(merged == False):
            merged = self.merge_point_to_o_cluster(x,self.o_micro_cluster)
        if(merged == False):
            new_cluster = self.MicroCluster(lembda=self.lembda)
            new_cluster.insert(x,self.current_timestamp)
            self.o_micro_cluster.append(new_cluster)


        # maintenance every Tp




    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X :

        Returns
        -------
        y :

        """
        y_pred=None
        return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.
           Parameters
           ----------
           
           Returns
           -------
            
           """
        proba=None
        return proba
