import numpy as np
import matplotlib.pyplot as plt

# calcular diferentes normas
def calculate_norms (X,Xi,norma,cov_i):
  if norma=='coseno':
      num=np.multiply(X,Xi).sum(axis=1)
      den=np.multiply(np.linalg.norm(X,axis=1,ord=2), np.linalg.norm(Xi,ord=2))
      return 1-num/den
  if norma=='mahalanobis': 
      delta=X-Xi
      return np.multiply(np.matmul(delta,cov_i),delta).sum(axis=1)
  if norma=='manhattan': p=1
  if norma=='euclidea': p=2
  if norma=='Lp': p=p
  return (abs(X-Xi)**p).sum(axis=1)**(1/p)

def get_distance_matrix(X,Y,cov_i,norms):
  D=np.zeros(shape=(len(norms),len(X),len(Y)))
  for i,norm in enumerate(norms):
    D[i]=np.array([calculate_norms(X,xi,norm,cov_i) for xi in Y]).reshape(len(X),-1)
  return D

## visualizar las distancias
def plot_distances(D,norms,png_name):
  nrows=2; ncols=2
  fig=plt.figure(figsize=(9,7))
  for i in range(len(norms)):
      ax=plt.subplot(nrows,ncols,i+1)
      cax=ax.matshow(D[i], cmap=plt.cm.Blues.reversed(), aspect='auto')
      ax.set_title(norms[i])
      fig.colorbar(cax, ax=ax,fraction=0.046)
  plt.tight_layout(); fig.savefig(png_name)