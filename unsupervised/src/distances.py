import numpy as np
import matplotlib.pyplot as plt

# calcular diferentes normas
def calculate_norms (X,Y,norma='euclidea',cov_i=''):
    if norma=='coseno':
        num=np.matmul(X,Y)
        den=np.linalg.norm(X,ord=2)*np.linalg.norm(Y,ord=2)
        return 1-num/den
    if norma=='mahalanobis': return np.sqrt((X-Y).dot(cov_i).dot((X-Y).T))
    if norma=='manhattan': p=1
    if norma=='euclidea': p=2
    if norma=='Lp': p=p
    return (abs(X-Y)**p).sum()**(1/p)

def get_distance_matrix(X,Y,cov_i,norms):
  D=np.zeros(shape=(len(norms),len(X),len(Y)))
  for i,norm in enumerate(norms):
    D[i]=np.array([calculate_norms(x,y,norma=norm,cov_i=cov_i) for x in X for y in Y]).reshape(len(X),-1)
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