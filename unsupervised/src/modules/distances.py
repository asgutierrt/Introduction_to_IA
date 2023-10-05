import numpy as np
import matplotlib.pyplot as plt

def covarianza_inversa(X):
    """ Calculates the inverse covariance matrix for a dataset."""
    return np.linalg.pinv(np.cov(X,rowvar=False)) 

def calculate_norms (X,Xi,norma,cov_i):
    """
    Calculates different norms between two data points.

    Args:
        X (numpy.ndarray): Data points.
        Xi (numpy.ndarray): Data point to compare each point in X.
        norma (str): The norm to calculate.
        cov_i (numpy.ndarray): The covariance matrix for the data.

    Returns:
        D (numpy.ndarray): The norm between the data points and Xi.
    """
    if norma=='coseno':
        num=np.multiply(X,Xi).sum(axis=1)
        den=np.multiply(np.linalg.norm(X,axis=1,ord=2), np.linalg.norm(Xi,ord=2))
        return 1-num/den
    if norma=='mahalanobis': 
        delta=X-Xi
        return np.multiply(np.matmul(delta,cov_i),delta).sum(axis=1)
    if norma=='manhattan': p=1
    if norma=='euclidea': p=2
    if 'Lp' in norma: p=int(norma.split('=')[1])
    return (abs(X-Xi)**p).sum(axis=1)**(1/p)
  

def get_distance_matrix(X,Y,cov_i,norms):
  """
  Calculates the distance matrix between two sets of data points.

  Args:
      X (numpy.ndarray): The first set of data points.
      Y (numpy.ndarray): The second set of data points.
      cov_i (numpy.ndarray): The covariance matrix for the data.
      norms (list): The norms to calculate.

  Returns:
      D (numpy.ndarray): The distance matrix between the two sets of data points.
      rows are X points and columns are Y points. third axis is the norm.
  """
  D=np.zeros(shape=(len(norms),len(X),len(Y)))
  for i,norm in enumerate(norms):
    D[i]=np.array([calculate_norms(X,xi,norm,cov_i) for xi in Y]).reshape(len(X),-1)
  return D

def plot_distances(D,norms,png_name):
  """
  Creates html visualizations of the distance matrix calculated with different norms.
  """
  nrows=int(np.ceil(len(norms)/2)); ncols=2
  fig=plt.figure(figsize=(9,7))
  for i in range(len(norms)):
      ax=plt.subplot(nrows,ncols,i+1)
      cax=ax.matshow(D[i], cmap=plt.cm.Blues.reversed(), aspect='auto')
      ax.set_title(norms[i])
      fig.colorbar(cax, ax=ax,fraction=0.046)
  plt.tight_layout(); fig.savefig(png_name)