from sklearn.base import BaseEstimator, ClusterMixin
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
        self.weight_threshold = weight_threshold;
        self.lembda = lembda;  # lambda is a defined structure in python, i put 'e' instead of 'a'
        self.epsilon = epsilon;
        self.mu = mu;
        self.beta = beta;
        self.p_micro_cluster = p_micro_cluster;
        self.o_micro_cluster = o_micro_cluster;
        self.init_buffer = init_buffer;
        self.timestamp = timestamp;
        self.current_timestamp = current_timestamp;
        self.tp = math.round(1 / self.lembda * math.log((self.beta * self.mu) / (self.beta * self.mu - 1))) + 1

    def fit(self, X, Y):

    # initialisation
    # offline dbscan



    def partial_fit(self, x, y):

    # ajout d'instance
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

        return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.
           Parameters
           ----------
           
           Returns
           -------
            
           """

        return proba
