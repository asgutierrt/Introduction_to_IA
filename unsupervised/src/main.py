from modules.load_data import *
from modules.distances import *
from modules.cluster_algorithms import *
from modules.support_functions import *   

from os.path import join, dirname
from os import chdir, getcwd


def main(filename='iris'):
    """
    Runs the clustering algorithm on the specified dataset.

    Args:
        filename (str): The name of the dataset to use. Defaults to 'iris'.

    Returns:
        None
    """
    # set working directory
    project_wd=dirname(dirname(__file__))
    chdir(project_wd)

    # information paths
    filepath = join(project_wd,'data',filename)
    fig_path= join(project_wd,'reports','figures')
    results_path= join(project_wd,'reports','group_matrix')

    # load data
    X,_ = load_data(filepath)
    
    ## calculate distances
    normas=['euclidea','mahalanobis','coseno','manhattan','Lp=3']
    cov_i = covarianza_inversa(X) #covarianza inversa
    D = get_distance_matrix(X,X,cov_i,norms=normas)

    ## plot distances
    plot_distances(D,normas,join(fig_path,'distances_XX.png'))

    # clustering: cajitas and k-vecinos
    plot_dims=[0,1,2]
    for norma_i in range(len(normas)):
        # 1. naive clustering: boxes
        n_groups=3
        cluster_pipeline(X, D[norma_i], naive_boxes, results_path, fig_path, 
                     plot_dims, 'boxes_norm_%s'%normas[norma_i], n_groups=n_groups)

        # 2. naive clustering: k-nearest neighbors
        k_n=50
        cluster_pipeline(X, D[norma_i], naive_kn, results_path, fig_path, 
                     plot_dims, 'kn_norm_%s'%normas[norma_i], k_n=k_n)

if __name__=="__main__":
    #main(filename='water_potability.txt')
    main() #iris