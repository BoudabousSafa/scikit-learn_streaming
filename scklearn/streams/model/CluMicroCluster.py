import math as math
import numpy as np

class CluMicroCluster:
    """
    Implementation of the MicroCluster data structure for the Clustream algorithm
    Parameters
    ----------
    :parameter nb_points is the number of points in the cluster
    :parameter linear_sum is the linear sum of the points in the cluster.
    :parameter squared_sum is  the squared sum of all the points added to the cluster.
    :parameter linear_time_sum is  the linear sum of all the timestamps of points added to the cluster.
    :parameter squared_time_sum is  the squared sum of all the timestamps of points added to the cluster.
    :parameter update_timestamp is used to indicate the last update time of the cluster
    """
    
    def __init__(self, nb_points=0, linear_sum=None, squared_sum=None, update_timestamp=0, creation_time= 0):
        self.nb_points = nb_points
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.update_timestamp = update_timestamp
        self.creation_time = creation_time
	self.linear_time_sum = 0
	self.squared_time_sum = 0
    
    def get_center(self, current_timestamp):
        center = self.linear_sum / self.nb_points
        return center

    def get_creation_time(self):
        return self.creation_time
	
    def insert(self,x, current_timestamp):
        self.nb_points += 1
        self.update_timestamp = current_timestamp
        for i in range(x) :
		self.linear_sum[i] += x[i]
		self.squared_sum[i] += math.pow(x[i],2)
		self.linear_time_sum[i] += current_timestamp
		self.squared_time_sum[i] += math.pow(current_timestamp,2)
			
