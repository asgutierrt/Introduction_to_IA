from modules.load_data import *
from modules.distances import *
from modules.cluster_algorithms import *
from modules.visualizations import *

from os.path import join

filename = 'water_potability.txt'

# store info on report folder
fig_path= join('..','reports','figures')
results_path= join('..','reports','group_matrix')
filepath = join('..','data',filename)

def main(filename='iris'):
    # store info on report folder
    fig_path= join('..','reports','figures')
    results_path= join('..','reports','group_matrix')
    filepath = join('..','data',filename)

    # load data
    X,_,_ = load_data(filepath)

    ## calculate covariance matrix inverse
    cov_i = covarianza_inversa(X)
    
    ## calculate and plot distances
    normas=['euclidea','mahalanobis','coseno','manhattan']
    D = get_distance_matrix(X,X,cov_i,norms=normas)
    plot_distances(D,normas,join(fig_path,'distances_XX.png'))

    plot_dims=[0,1,2]
    # run clustering algorithms on each distance matrix
    for norma_i in range(len(normas)):
        # 1. naive clustering: boxes
        n_groups=3

        ## create grouping matrix
        G0, ref_points = naive_boxes(X,D,n_groups,norma_i)

        ## save the results
        save_results(G0,join(results_path,'boxes_norm_%s.txt'%normas[norma_i]))

        ## display the results
        fig_name=join(fig_path,'boxes_norm_%s.html'%normas[norma_i])
        plot_clusters(X,G0,plot_dims,ref_points,normas[norma_i],fig_name)
        

        # 2. naive clustering: k-nearest neighbors
        k_n=50

        ## create grouping matrix
        G0, ref_points = naive_kn(X,D,k_n,norma_i)

        ## save the results
        save_results(G0,join(results_path,'kn_norm_%s.txt'%normas[norma_i]))

        ## display the results
        fig_name=join(fig_path,'kn_norm_%s.html'%normas[norma_i])
        plot_clusters(X,G0,plot_dims,ref_points,normas[norma_i],fig_name)

if __name__=="__main__":
    main(filepath='iris')