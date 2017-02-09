from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scklearn.streams.model import MicroCluster as model
import numpy as np
import math as math


class DenStream(BaseEstimator, ClusterMixin):
    """Implementation of DenStream
       Read more in the :

       Parameters
       ----------

       Examples
       --------
       * >>> X =
       * >>> y =
       * >>> from scklearn.xxx import DenStream
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
                 epsilon=0.01, mu=10, beta=0.2, p_micro_cluster=[], o_micro_cluster=[], init_buffer=None,
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
        s = (self.beta * self.mu) / (self.beta * self.mu - 1)
        self.tp = math.ceil(1 / self.lembda * math.log2(s)) + 1

    def fit(self, X, Y=None):
        # initialisation
        # offline dbscan
        X = check_array(X, accept_sparse='csr')
        nb_initial_points = X.shape[0]
        if nb_initial_points >= self.init_points_option:
            dbscan = DBSCAN()
            m_cluster_labels = dbscan.fit_predict(X, Y)
            X = np.column_stack((m_cluster_labels, X))
            initial_clusters = [X[X[:, 0] == l][:, 1:] for l in set(m_cluster_labels) if
                                l.astype(str) != '-1']
            [self.p_micro_cluster.append(self.create_micro_cluster(cluster)) for cluster in initial_clusters]

    def create_micro_cluster(self, cluster):
        linear_sum = np.zeros(cluster.shape[1])
        squared_sum = np.zeros(cluster.shape[1])
        new_m_cluster = model.MicroCluster(nb_points=0, linear_sum=linear_sum, squared_sum=squared_sum,
                                           update_timestamp=0, lembda=self.lembda)
        [new_m_cluster.insert(point, self.current_timestamp) for point in cluster]
        return new_m_cluster

    def merge_point_to_p_cluster(self, x, clusters):
        cluster_centers = list(map((lambda i: i.get_center(self.current_timestamp)), clusters))
        # Apply scikit learn nearest neignbor function over p_micro_cluster_centers in order to get the nearest potential micro-cluster
        neighbors_algorithm = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cluster_centers)

        nearest_clusters = neighbors_algorithm.kneighbors(x.reshape(1,-1), 1, return_distance=False)
        cluster_index = nearest_clusters[0][0] if len(nearest_clusters)>0 else 0

        # self.candidate_cluster = filter(lambda cp: cp.getCenter()== self.nearest_p_micro_cluster, self.p_micro_cluster)
        e_cluster = self.p_micro_cluster[cluster_index]
        e_cluster.insert(x, self.current_timestamp)
        if e_cluster.get_radius(self.current_timestamp) <= self.epsilon:
            self.p_micro_cluster[cluster_index].insert(x, self.current_timestamp)
            return True
        else:
            return False

    def merge_point_to_o_cluster(self, x, clusters):
        cluster_centers = list(map((lambda i: i.get_center(self.current_timestamp)), clusters))
        # Apply scikit learn nearest neignbor function over p_micro_cluster_centers in order to get the nearest potential micro-cluster
        neighbors_algorithm = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(cluster_centers)
        nearest_clusters = neighbors_algorithm.kneighbors(x.reshape(1,-1), 1, return_distance=False)[0]
        cluster_index = nearest_clusters[0] if len(nearest_clusters) > 0 else 0
        e_cluster = self.o_micro_cluster[cluster_index]
        e_cluster.insert(x, self.current_timestamp)
        if e_cluster.get_radius(self.current_timestamp) <= self.epsilon:
            self.o_micro_cluster[cluster_index].insert(x, self.current_timestamp)
            if (self.o_micro_cluster[cluster_index].get_weight(self.current_timestamp) > self.beta * self.mu):
                self.p_micro_cluster.append(self.o_micro_cluster[cluster_index])
                del self.o_micro_cluster[cluster_index]
            return True
        else:
            return False

    def partial_fit(self, x, y):
        self.timestamp += 1
        self.current_timestamp = self.timestamp
        # ajout d'instance
        merged = self.merge_point_to_p_cluster(x[0,:], self.p_micro_cluster)
        if (merged == False and len(self.o_micro_cluster) > 0):
            merged = self.merge_point_to_o_cluster(x[0,:], self.o_micro_cluster)
        if (merged == False):
            new_cluster = self.create_micro_cluster(np.reshape(x[0,:], (1, len(x[0,:]))))
            self.o_micro_cluster.append(new_cluster)

        # maintenance every Tp
        if (self.timestamp % self.tp == 0):
            removeList = [i for i in range(len(self.p_micro_cluster)) if self.p_micro_cluster[i].get_weight(self.current_timestamp) < self.beta * self.mu]

            np.delete(self.p_micro_cluster, removeList, axis=0)

            removeList =[]
            for i in range(len(self.o_micro_cluster)):
                c = self.o_micro_cluster[i]
                t0 = c.get_creation_time()
                xsi1 = math.pow(2, -(self.lembda * (self.current_timestamp - t0 + self.tp))) - 1
                xsi2 = math.pow(2, -(self.lembda * self.tp)) - 1
                xsi = xsi1 / xsi2
                if c.get_weight(self.current_timestamp) < xsi:
                    removeList.append(i)

            np.delete(self.o_micro_cluster, removeList, axis=0)

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X :

        Returns
        -------
        y :

        """
        cluster_centers = list(map((lambda i: i.get_center(self.current_timestamp)), self.p_micro_cluster))
        dbscan = DBSCAN(eps=2 * self.epsilon, min_samples=self.min_points)
        result = dbscan.fit_predict(cluster_centers, None)
        return result
