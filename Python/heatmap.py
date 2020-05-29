import numpy as np

def initMap(dimension):
    map = np.zeros((dimension, dimension))
    map[0,:] = 1
    map[:,0] = 1
    return map
