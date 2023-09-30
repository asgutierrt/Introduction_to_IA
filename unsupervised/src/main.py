from modules.load_data import *
from modules.distances import *
from modules.cluster_algorithms import *
from modules.visualizations import *


def main(filepath='iris'):
    # get this file's path
    fig_path='reports/figures/'
    results_path='reports/'

    # load data
    X,_,_ = load_data(filepath)
    ## calculate covariance matrix inverse
    cov_i = covarianza_inversa(X)
    ## calculate and plot distances
    normas=['euclidea','mahalanobis','coseno','manhattan']
    D = get_distance_matrix(X,X,cov_i,norms=normas)
    plot_distances(D,normas,fig_path+'distances_XX.png')


    # naive clustering: boxes
    norma_i=0; n_groups=3
    G0, ref_points = naive_boxes(X,D,n_groups,norma_i)
    ## save the results
    save_results(G0,results_path+'boxes_norm_%s.txt'%normas[norma_i])
    ## display the results
    fig_name=fig_path+'boxes_norm_%s.html'%normas[norma_i]
    plot_clusters(X,G0,[0,1,2],ref_points,normas[norma_i],fig_name)
    

    # naive clustering: k-nearest neighbors
    k_n=50; norma_i=0
    G0, ref_points = naive_kn(X,D,k_n,norma_i)
    ## save the results
    save_results(G0,results_path+'kn_norm_%s.txt'%normas[norma_i])
    ## display the results
    fig_name=fig_path+'kn_norm_%s.html'%normas[norma_i]
    plot_clusters(X,G0,[0,1,2],ref_points,normas[norma_i],fig_name)

if __name__=="__main__":
    main(filepath='iris')