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
    
    def __init__(self, nb_points=0, linear_sum=0, squared_sum=0, linear_time_sum=0, squared_time_sum=0,
		 update_timestamp=0, creation_time= 0, max_num_kernels=100, kernel_radifactor=2):
        self.nb_points = nb_points
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.update_timestamp = update_timestamp
        self.creation_time = creation_time
	self.linear_time_sum = linear_time_sum
	self.squared_time_sum = update_timestamp
	self.m = max_num_kernels
	self.t = kernel_radifactor
	self.epsilon = 0.00005
	self.min_variance = math.exp(-50)
	self.radius_factor = 1.8

    def insert(self,x, current_timestamp):
        self.nb_points += 1
        self.update_timestamp = current_timestamp
        for i in range(x) :
            self.linear_sum[i] += x[i]
            self.squared_sum[i] += math.pow(x[i],2)
	    self.linear_time_sum[i] += current_timestamp
	    self.squared_time_sum[i] += math.pow(current_timestamp,2)

    def get_relevanceStamp(self):
	if(self.nb_points < 2*self.m):
		return self.get_mutime()
	return self.get_mutime()

    def get_mutime(self):
	return self.linear_time_sum/self.nb_points

    def get_sigmatime(self):
	return math.sqrt(self.square_time_sum/self.nb_points - math.pow((self.linear_time_sum/self.nb_points),2))

    def get_quantile(self,x):
	#TODO hold the exception ( x >= 0 && x <= 1 )
	return math.sqrt( 2 ) * self.inverse_error( 2*x - 1 )
	
    def get_radius(self):
	if(N == 1): 
		return 0
	return self.get_deviation()*self.radius_factor

   def get_deviation(self):
	variance = self.get_variance_vector()
        sum_deviation = 0
        for i in range(len(variance)):
            d = math.sqrt(variance[i]);
            sum_deviation += d
        return sum_deviation / len(variance)			
