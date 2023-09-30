import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
 
def encode_array (x):
  enc = OneHotEncoder(handle_unknown='ignore')
  return enc.fit_transform(x.reshape((-1,1))).toarray()

def load_data(filepath):
    # load data
    if filepath.split(".")[-1]=='json':
        df = pd.read_json(filepath, orient="table")
    elif filepath.split(".")[-1]=='txt':
        df = pd.read_csv(filepath, sep='\t')
    elif filepath.split(".")[-1]=='csv':
        df = pd.read_csv(filepath)
    else: #'iris'
        iris = datasets.load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])  
    
    # drop columns that are more than 40% empty
    empty_tol=0.4
    df.dropna(axis=1,thresh=df.shape[0]*(1-empty_tol),inplace=True) 
    # drop datapoints that have missing values
    df.dropna(axis=0,inplace=True)
    
    # hot encode cathegorical columns
    encode_cols=[] # ['Potability'] ['target']
    df_encoded = pd.get_dummies(df, columns=encode_cols)

    # get data
    X = df_encoded.iloc[:,:-1].to_numpy()

    # normalize data
    X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))
    m=len(X[0]); N=len(X)
    return X, N, m

def covarianza_inversa(X):
    cov=np.cov(X,rowvar=False)
    cov_i=np.linalg.pinv(cov) 
    return cov_i

def save_results(array,file_name):
    np.savetxt(file_name, array, delimiter=",", fmt='%i')
    