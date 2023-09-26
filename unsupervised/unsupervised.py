# instalar e importar las librerias necesarias
try:
    # data
    from sklearn import datasets
    import numpy as np

    # plotting
    import matplotlib.pyplot as plt
    from babyplots import Babyplot

except: # instalar librerias
    import os
    cmds=['pip install --upgrade pip', 'pip install scikit-learn numpy matplotlib babyplots']
    for cmd in cmds: os.system(cmd)

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

if __name__ == '__main__':
    # cargar el dataset
    iris = datasets.load_iris()
    X = iris.data; y = iris.target
    X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)) # normalizar
    cov=np.cov(X,rowvar=False); cov_i=np.linalg.pinv(cov) # covarianza

    # plot distances
    norms=['euclidea','mahalanobis','coseno','manhattan']

    nrows=2; ncols=2
    fig=plt.figure(figsize=(9,7))

    for i in range(len(norms)):
        D=np.array([norma(x,y,norma=norms[i],cov_i=cov_i) for x in X for y in X]).reshape(len(X),-1)
        ax=plt.subplot(nrows,ncols,i+1)
        cax=ax.matshow(D, cmap=plt.cm.Blues.reversed())
        ax.set_title(norms[i])
        fig.colorbar(cax, ax=ax,fraction=0.046)
    fig.tight_layout(); fig.savefig('distances_XX.png')

    # naive clustering
    n_groups=3
    func = lambda x: x/((x.max()-x.min())/n_groups)
    G=np.apply_along_axis(func, 0, D).astype(int)

    # plot naive clustering
    x_ref=0
    bp = Babyplot(background_color="#ffffddff", turntable=True)
    bp.add_plot(X, "pointCloud", "categories", G[x_ref], {"colorScale": "Dark2"})
    bp
