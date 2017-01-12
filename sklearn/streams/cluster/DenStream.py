import numpy as np
from scipy import stats



class DenStream(parameters):
     """Implementation of DenStream
        Read more in the :
        
        Parameters
        ----------
              
        Examples
        --------
        >>> X = 
        >>> y = 
        >>> from sklearn.xxx import DenStream
        >>> ds = DenStream(params)
        >>> ds.fit(X, y) # doctest: +ELLIPSIS
        DenStream(...)
        >>> print(ds.predict())
       
        See also
        --------
       
        Notes
        -----
        
        """
    def __init__(self, ):
        self.h = 1000 # horizon : Range of the window.
        self.minPoints = 10 # Minimal number of points cluster has to contain.
        self.initPointsOption = 1000 # Number of points to use for initialization."
        self.weightThreshold = 0.01;
        self.lembda; #lambda is a defined structure in python, i put 'e' instead of 'a'
        self.epsilon = 0.01;
        self.minPoints;
        self.mu = 1;
        self.beta = 0.001;
        self.p_micro_cluster;
        self.o_micro_cluster;
        self.initBuffer;
        self.initialized;
        self.timestamp = 0;
        self.currentTimestamp;
        self.tp;
    
    
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
