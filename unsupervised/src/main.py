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
    # set working directory
    project_wd=dirname(dirname(__file__)); chdir(project_wd)

    # information paths
    figs_path= join(project_wd,'reports','figures')
    results_path= join(project_wd,'reports','group_matrix')

    # load data
    X = load_data(join(project_wd,'data',filename),verbose=False)
    X_expanded = encode(X, encoding_layers_dims=[15,20], verbose=False)
    X_embeded = umap.UMAP().fit_transform(X) # 2D embedding using UMAP

    # global setup for all clustering algorithms
    normas=['euclidea','mahalanobis','coseno','manhattan','Lp=3']
    
    for X, X_name in zip([X, X_expanded, X_embeded],['x_original','encoded','embeded']):
        print('processing %s'%X_name)
        plot_dims=list(range(min(3,len(X[0])))) # dimensions to plot

        # distance matrix
        cov_i = covarianza_inversa(X)
        D_XToX = get_distance_matrix(X,X,cov_i,norms=normas)
        plot_distances(D_XToX,normas,join(figs_path,'distances_%s.png'%X_name))

        grid = make_grid(*X.shape)
        cov_i_grid = covarianza_inversa(grid)
        D_GridToX = get_distance_matrix(grid,X,cov_i,norms=normas)
        D_GridToGrid = get_distance_matrix(grid,grid,cov_i_grid,norms=normas)

        # cluster problem object
        cluster_setup=ClusterProblem(X_name,normas,results_path,figs_path,plot_dims)

        # 1. naive clustering: cajitas
        n_groups=3
        cluster_setup.do_ClusterPipeline(X, D_XToX, naive_boxes, 'naive_boxes', n_groupps=n_groups)

        # 2. naive clustering: k-vecinos
        k_n=50
        cluster_setup.do_ClusterPipeline(X, D_XToX, naive_kn, 'naive_vecinos', k_n=k_n)
        
        # 3. mountain clustering (with grid data)
        ra=0.3
        stop_criteria='k_centroids' # or 'low_density' and epsilon [=0.2] argument
        k=5
        cluster_setup.do_ClusterPipeline(X, D_XToX**2, density_substraction, 'substractive_alg', 
                                                 ra=ra, kind='substractive',k=k,stop_criteria=stop_criteria)

        # 4. substractive clustering 
        ra=0.3
        stop_criteria='k_centroids' # or 'low_density' and epsilon [=0.2] argument
        k=5                                      
        cluster_setup.do_ClusterPipeline(X, zip(D_GridToX**2,D_GridToGrid**2), density_substraction, 
                                                 'mountain_alg', ra=ra, kind='mountain',k=k,stop_criteria=stop_criteria)

if __name__=="__main__":
    #main(filename='water_potability.txt')
    main() #iris 