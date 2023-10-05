from numpy.ma import masked_array as masked_array
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from itertools import product


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

def save_results(array,array_names,file_name):
    """ Saves a numpy.ndarray to a file. """
    with open(file_name,'w') as f:
        for sub_array, sub_array_name in zip(array,array_names):
            f.write('with %s distance:\n'%sub_array_name)
            np.savetxt(f, sub_array, fmt='%.2f', delimiter=',')
            f.write('\n')


def make_grid(N,m,**kwargs):
    """ Makes a grid of points in the unit square. """
    malla=lambda m, n_intervals: np.array(list(product(np.arange(n_intervals +1)*1/n_intervals,repeat=m)))
    n_grid_intervals = lambda N,m: int(np.exp(np.log(N)/m)-1)
    n_intervals=kwargs['n_intervals'] if 'n_intervals' in kwargs else min(2,n_grid_intervals(N,m))
    return malla(m,n_intervals)