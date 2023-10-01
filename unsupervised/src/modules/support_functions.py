from numpy.ma import masked_array as masked_array
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def threshold_distance(x,n_groups):
    """
    Computes a threshold distance for the given array of distances.
    """
    return x/((x.max()+10**-10-x.min())/n_groups)

def unclassified_points(points,G0):
    """
    Masks the points that have already been classified in the grouping matrix.
    """
    return masked_array(points, (G0.sum(axis=1)!=0))

def encode_array (x):
  """ One-hot encodes a 1D array. """
  enc = OneHotEncoder(handle_unknown='ignore')
  return enc.fit_transform(x.reshape((-1,1))).toarray()

def save_results(array,file_name):
    """ Saves a numpy.ndarray to a file. """
    np.savetxt(file_name, array, delimiter=",", fmt='%i')