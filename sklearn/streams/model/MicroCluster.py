import math as math

import numpy as np


class MicroCluster:
    """
    Implementation of the MicroCluster data structure for the Denstream algorithm

    Parameters
    ----------
    :parameter nb_points is the number of points in the cluster
    :parameter linear_sum is the linear sum of the points in the cluster.
    :parameter squared_sum is  the squared sum of all the points added to the cluster.
    :parameter update_timestamp is used to indicate the last update time of the cluster
    :parameter lembda is Denstream classifier parameter that allows to adjust the importance associated to historical
    data.It's used in MicroCluster level to compute the fadinf function necessary in updating th Micro cluster center,
     weight and radius
    """

    def __init__(self, nb_points=0, linear_sum=None, squared_sum=None, update_timestamp=0, lembda=0.01, creation_time= 0):
        self.nb_points = nb_points
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.update_timestamp = update_timestamp
        self.lembda = lembda
        self.creation_time = creation_time

    def get_center(self, current_timestamp):
        fading_coef = self.fading_function(current_timestamp)
        weight = self.get_weight(fading_coef)
        center = [fading_coef / weight * self.linear_sum[i] for i in range(len(self.linear_sum))]
        return center

    def get_weight(self, current_timestamp):
        fading_coef = self.fading_function(current_timestamp)
        weight = self.nb_points * fading_coef
        return weight

    def get_radius(self, current_timestamp):
        fading_coef = self.fading_function(current_timestamp)
        weight = self.get_weight(fading_coef)
        radius = [self.compute_radius(i, fading_coef, weight) for i in range(len(self.linear_sum))]
        return radius

    def compute_radius(self, index, fading_coef, weight):
        x1 = fading_coef / weight * self.squared_sum[index]
        x2 = (fading_coef / weight * self.linear_sum[index])
        return math.sqrt(x1 - math.pow(x2, 2))

    def fading_function(self, current_timestamp):
        delta = current_timestamp - self.update_timestamp
        return math.pow(2, -(self.lembda * delta))

    def get_creation_time(self):
        return self.creation_time

    def insert(self,x, current_timestamp):
        self.nb_points += 1
        self.update_timestamp = current_timestamp
        for i in range(x) :
            self.linear_sum[i] += x[i]
            self.squared_sum[i] += math.pow(x[i],2)

