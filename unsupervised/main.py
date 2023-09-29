from src.load_data import *
from src.distances import *
from src.cluster_algorithms import *
from src.visualizations import *

import os

if __name__=="__main__":
    # get this file's path
    curr_path=os.path.dirname(os.path.abspath(__file__))
    fig_path=curr_path+'/results/'

    # load data
    X, N, m = load_data()
    cov_i = covarianza_inversa(X)

    # calculate and plot distances
    normas=['euclidea','mahalanobis','coseno','manhattan']
    D = get_distance_matrix(X,X,cov_i,norms=normas)
    plot_distances(D,normas,fig_path+'distances_XX.png')

    # naive clustering: single reference point
    norma_i=0; n_groups=3
    G=naive(D,n_groups,norma_i)

    ## pick one reference point and encode its cluster
    x_ref=0
    fig_name=fig_path+'naive_norm_%s.html'%normas[norma_i]
    fig=plot_clusters(X,encode_array(G[x_ref]),[0,1,2],[x_ref],normas[norma_i],fig_name)

    ## visualize multiple classifications
    x_references=[0,50,25]
    fig_name=fig_path+'multiple_naive_norm_%s.html'%normas[norma_i]
    fig=subplots(X,[encode_array(G[x]) for x in x_references],[0,1,2],
                 [[x] for x in x_references],normas[norma_i],fig_name)

    # naive clustering: adaptative reference point
    norma_i=0; n_groups=3
    G0, ref_points = naive_boxes(X,D,n_groups,norma_i)
    fig_name=fig_path+'naive_boxes_norm_%s.html'%normas[norma_i]
    fig=plot_clusters(X,G0,[0,1,2],ref_points,normas[norma_i],fig_name)

    # naive clustering: k-nearest neighbors
    k_n=50; norma_i=0
    G0, ref_points = naive_kn(X,D,k_n,norma_i)
    fig_name=fig_path+'naive_kn_norm_%s.html'%normas[norma_i]
    fig=plot_clusters(X,G0,[0,1,2],ref_points,normas[norma_i],fig_name)