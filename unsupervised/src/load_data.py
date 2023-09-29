import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

def encode_array (x):
  enc = OneHotEncoder(handle_unknown='ignore')
  return enc.fit_transform(x.reshape((-1,1))).toarray()

def load_data(path=''):
    if path!='': 
        print('load file')
    else: # load iris dataset
        iris = datasets.load_iris()
        X = iris.data
        # y = iris.target # clases
    
    # normalizar los datos
    X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0)) # normalizar
    m=len(X[0]); N=len(X)
    return X, N, m

def covarianza_inversa(X):
    cov=np.cov(X,rowvar=False)
    cov_i=np.linalg.pinv(cov) 
    return cov_i




    