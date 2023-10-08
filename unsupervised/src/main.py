from modules.load_data import *
from modules.distances import *
from modules.cluster_algorithms import *
from modules.support_functions import *   
from modules.autoencoder import encode
import umap

from os.path import join, dirname
from os import chdir


def main(filename='iris'):
    """
    Runs the clustering algorithm on the specified dataset.

    Args:
        filename (str): The name of the dataset to use. Defaults to 'iris'.
    """
    # 1. set working directory and paths
    project_wd=dirname(dirname(__file__)); chdir(project_wd)
    figs_path= join(project_wd,'reports','figures')
    results_path= join(project_wd,'reports','group_matrix')

    # 2. load and process data
    X_name='original'
    X = load_data(join(project_wd,'data',filename),verbose=True)
    if X_name=='encoded':
        X = encode(X, encoding_layers_dims=[int(X.shape[1]*1.2),int(X.shape[1]*1.5)], verbose=True)
    elif X_name=='umap':
        print('calculating umap embedding')
        X = umap.UMAP().fit_transform(X)

    # 3. global setup for all clustering algorithms
    print('calculating distance matrix')
    ## distance matrix
    normas=['euclidea','mahalanobis','coseno','manhattan','Lp=3']
    cov_i = covarianza_inversa(X)
    D_XToX = get_distance_matrix(X,X,cov_i,norms=normas)
    plot_distances(D_XToX,normas,join(figs_path,'distances_%s.png'%X_name))
    ## cluster problem object
    cluster_setup=ClusterProblem(X_name,normas,results_path,figs_path)

    print('starting clustering algorithms')

    # 1. naive clustering: cajitas
    n_groups=3
    cluster_setup.do_ClusterPipeline(X, D_XToX, naive_boxes, 'naive_boxes', n_groupps=n_groups)

    # 2. naive clustering: k-vecinos
    k_n=50
    cluster_setup.do_ClusterPipeline(X, D_XToX, naive_kn, 'naive_vecinos', k_n=k_n)
    
    # 3. mountain clustering (with grid data)
    grid = make_grid(X,n_intervals=3)
    D_GridToX = get_distance_matrix(grid,X,cov_i,norms=normas)
    D_GridToGrid = get_distance_matrix(grid,grid,covarianza_inversa(grid),norms=normas)
    ra=0.3
    stop_criteria='k_centroids' # or 'low_density' and epsilon [=0.2] argument
    k=5                                      
    cluster_setup.do_ClusterPipeline(X, zip(D_GridToX**2,D_GridToGrid**2), density_substraction, 'mountain_alg', 
                                    ra=ra,kind='mountain',stop_criteria=stop_criteria,k=k)
    
    # 4. substractive clustering 
    ra=0.3
    stop_criteria='k_centroids' # or 'low_density' and epsilon [=0.2] argument
    k=5
    cluster_setup.do_ClusterPipeline(X, D_XToX**2, density_substraction, 'substractive_alg', 
                                    ra=ra, kind='substractive',stop_criteria=stop_criteria,k=k)

    print('clustering algorithms finished')
    
if __name__=="__main__":
    # main(filename='water_potability.txt')
    main() #iris 