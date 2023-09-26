# instalar e importar las librerias necesarias
try:
    # data
    from sklearn import datasets
    import numpy as np
    from itertools import product

    # plotting
    import matplotlib.pyplot as plt
    from babyplots import Babyplot

except: # instalar librerias
    import os
    cmds=['pip install --upgrade pip', 'pip install scikit-learn numpy matplotlib babyplots itertools']
    for cmd in cmds: os.system(cmd)
    print('libraries installed, please run again')

def norma (X,Y,norma='euclidea',cov_i=''):
  if norma=='coseno':
    num=np.matmul(X,Y)
    den=np.linalg.norm(X,ord=2)*np.linalg.norm(Y,ord=2)
    return 1-num/den
  if norma=='mahalanobis':
    return np.sqrt((X-Y).dot(cov_i).dot((X-Y).T))
  if norma=='manhattan': p=1
  if norma=='euclidea': p=2
  if norma=='Lp': p=p
  return (abs(X-Y)**p).sum()**(1/p)

def plot_distances(X, Y, cov_i, name, figsize=(9,7)):
    norms=['euclidea','mahalanobis','coseno','manhattan']
    nrows=2; ncols=2
    fig=plt.figure(figsize=figsize)

    for i in range(len(norms)):
        D=np.array([norma(x,y,norma=norms[i],cov_i=cov_i) for x in X for y in Y]).reshape(len(X),-1)
        ax=plt.subplot(nrows,ncols,i+1)
        cax=ax.matshow(D, cmap=plt.cm.Blues.reversed())
        ax.set_title(norms[i])
        fig.colorbar(cax, ax=ax,fraction=0.046)
    fig.tight_layout(); fig.savefig(name)
    return D # last D

if __name__ == '__main__':
    # cargar el dataset
    iris = datasets.load_iris()
    X = iris.data; y = iris.target
    X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)) # normalizar
    m=len(X[0]); N=len(X)
    cov=np.cov(X,rowvar=False); cov_i=np.linalg.pinv(cov) # covarianza

    # plot distances
    D = plot_distances(X, X, cov_i, 'distances_XX.png')

    # naive clustering
    n_groups=3
    func = lambda x: x/((x.max()-x.min())/n_groups)
    G=np.apply_along_axis(func, 0, D).astype(int)

    # plot naive clustering
    x_ref=0
    dims=range(0,3)
    #bp = Babyplot()
    #bp.add_plot(X[:,dims], "pointCloud", "categories", G[x_ref], {"colorScale": "Paired"})
    #bp.save_as_html('groups.html') # abrir en el navegador

    # make grid
    malla=lambda m, n_intervals: list(product(np.arange(n_intervals +1)*1/n_intervals,repeat=m))
    n_grid_intervals= lambda N,m: int(np.exp(np.log(N)/m)-1)
    grid=malla(m,n_grid_intervals(N,m))

    # plot distances
    D = plot_distances(X, grid, cov_i, 'distances_vX.png', figsize=(7,10))
