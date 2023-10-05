import pandas as pd
import numpy as np
from sklearn import datasets

def load_data(filepath, verbose=False, **kwargs):
    """
    Args:
        filepath (str): supports .json, .txt, and .csv files.
             
        **kwargs: 
            - empty_tol (float): The tolerance percentage for empty values per column. 
                                Default value (0.4). If a column has more than 40% of empty values, it will be dropped.
            - encode_cols (list): A list of column names to one-hot encode. Default is an empty list.

    Returns:
        X (numpy.ndarray): Each row is a data point and each column is a feature.
        shape (tuple): The dimensions of the data.
    """
    empty_tol=kwargs['empty_tol'] if 'empty_tol' in kwargs else 0.4
    encode_cols=kwargs['encode_cols'] if 'encode_cols' in kwargs else []
 
    # read file
    df = read_file (filepath)

    # do an exploratory analysis
    if verbose:
        explore_data(df,verbose=verbose)
    
    ## deal with missing values
    # drop columns that are more than 40% empty
    df.dropna(axis=1,thresh=df.shape[0]*(1-empty_tol),inplace=True) 
    # drop datapoints that have missing values
    df.dropna(axis=0,inplace=True)
    
    ## deal with cathegorical columns
    df_encoded = pd.get_dummies(df, columns=encode_cols)

    ## to numpy
    X = df_encoded.iloc[:,:-1].to_numpy()
    #y = df_encoded.iloc[:,-1].to_numpy()

    ## normalize features
    X = (X-np.min(X,axis=0))/(np.max(X,axis=0)-np.min(X,axis=0))*2-1
    return X

def explore_data(df, verbose=False):
    ''' 
    1. number of ocurrences of each class: check for imbalanced data
    2. basic statistics of the data: 
        - check for (very) different scales which is a problem for distance based algorithms.
        either way, the data is normalized in the pipeline.
        - nan values: count missing values in each feature
    '''
    print("exploration of data:\n")
    print("1. number of ocurrences of each class: check for imbalanced data")
    labels=df.iloc[:,-1].value_counts().reset_index().rename(columns={0: "count"}).set_index(df.columns[-1])
    labels['%']=labels['count']/labels['count'].sum()*100
    print(labels.to_string()+"\n")
    print("2. basic statistics on the data: check scales of the values and nan value count\n")
    df_describe=df.describe()
    df_describe.loc['scale']=df_describe.loc['mean'].apply(lambda x: 'e'+('%e'%x).split('e')[1])
    df_describe.loc['nan count']=df.isna().sum()
    print(df_describe.to_string())
    return df_describe


def read_file (filepath):
    """ Reads file in filepath and returns a pandas dataframe.

    Notes:
         - Each row is a data point and each column is a feature. last column is the label.
         - the project's data folder contains an example of each file type and structure.
         - If the file extension is not recognized, it loads the iris dataset.
    """

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
    return df