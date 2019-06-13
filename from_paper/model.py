import numpy as np
from util import as_row_matrix
from subspace import pca , lda , fisherfaces , project
from distance import EuclideanDistance


class BaseModel ( object ):

    def __init__ ( self , X= None , y= None , dist_metric = EuclideanDistance () , num_components=0) :
        self . dist_metric = dist_metric
        self . num_components = 0
        self . projections = []
        self .W = []
        self . mu = []
        if (X is not None ) and (y is not None ):
            self . compute (X ,y )


    def compute ( self , X , y):
        raise NotImplementedError (" Every BaseModel must implement the compute method .")


    def predict ( self , X):
    minDist = np . finfo (’float ’) . max
    minClass = -1
    Q = project ( self .W , X. reshape (1 , -1) , self . mu )
    for i in xrange ( len ( self . projections )):
        dist = self . dist_metric ( self . projections [ i], Q)
        if dist < minDist :
            minDist = dist
            minClass = self . y[i]
    return minClass
