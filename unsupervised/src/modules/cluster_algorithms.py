import numpy as np
from modules.support_functions import *
from os.path import join
from modules.visualizations import *

"""
This module contains implementations of various clustering algorithms.

Args:
        X (numpy.ndarray): The data to cluster.
        D (numpy.ndarray): The distance matrix for the data.
        **kwargs: Additional keyword arguments to pass to each function.

Returns:
    G (numpy.ndarray): The grouping matrix for the clustered data.
    ref_points (list): The reference points used to group the data.

Functions:
- cluster_pipeline: computes, saves and plots the results of a specified clustering algorithm.
- naive: simple, naive_boxes, naive_kn
"""

def cluster_pipeline(X, D, cluster_func, 
                     results_path, fig_path, 
                     plot_dims, cluster_name, **kwargs):
    """
    Clusters the data using the specified algorithm.

    Args:
        X (numpy.ndarray): The data to cluster.
        D (numpy.ndarray): The distance matrix for the data.
        cluster_func (function): The clustering algorithm to use.
        results_path (str): The path to save the results.
        fig_path (str): The path to save the figures.
        plot_dims (list): The dimensions to plot.
        cluster_name (str): The name of the clustering algorithm.
        **kwargs: Additional arguments to pass to the clustering function.
    Returns:
        None
    """
    # create grouping matrix
    G0, ref_points = cluster_func(X, D, **kwargs)

    # save the results
    save_results(G0, join(results_path, cluster_name+'.txt'))

    # display the results
    fig_name = join(fig_path, cluster_name+'.html')
    return plot_clusters(X, G0, plot_dims, ref_points, cluster_name, fig_name)
     

def naive (_, D,**kwargs):
    """ naive one set per datapoint: groups according to a threshold distance. 
        **kwargs: 
            - n_groups (int): numero de grupos a formar. determina el calculo del valor threshold. default=3
            - x_ref (int): escoger solo la clasificacion dada por x_ref. La devuelve en formato hot-encoded.
                           default=None devuelve todas las posibles clasificaciones.
    """
    n_groups=kwargs['n_groups'] if 'n_groups' in kwargs else 3
    x_ref=kwargs['x_ref'] if 'x_ref' in kwargs else None

    G=np.apply_along_axis(threshold_distance, 1, D, n_groups=n_groups).astype(int)
    if x_ref is not None: 
        return encode_array(G[x_ref,:]), [x_ref]
    return G, list(range(len(G)))

def naive_boxes (X,D,**kwargs):
    '''
    Junta los resultados de naive en una unica clasificacion:
    Agrupa en un subconjunto a los puntos que estan a una distancia threshold a un punto de referencia. 
    Luego, toma un punto que no se ha clasificado como referencia y crea el siguiente subconjunto.
    Termina cuando todos los puntos han sido clasificados.
    
    **kwargs: 
        - n_groups (int): numero de grupos a formar. determina el calculo del valor threshold. default=3
    '''
    n_groups=kwargs['n_groups'] if 'n_groups' in kwargs else 3
    # clasifica segun una distancia threshold calculada con la minima y maxima distancia entre puntos en el dataset
    G_threshold=np.apply_along_axis(threshold_distance, 0, D.flatten(),
                                    n_groups=n_groups).reshape((len(X),-1))
    
    x_ref=0 # el primer subconjunto se forma con los puntos cerca a x_ref
    ref_points=[] # almacenar que punto de referencia han sido usados
    G0=np.zeros(shape=(len(X),n_groups)) # almacenar la clasificacion de cada punto

    for i in range(n_groups):
        # store current reference point
        ref_points.append(x_ref)

        # mark all points that are a threshold distance to x_ref
        mask=G_threshold[x_ref]<1
        G0[mask,i]=1

        # find the closest point to x_ref that is not yet classified
        # make it the next reference point
        x_ref=unclassified_points(G_threshold[x_ref],G0).argmin()
    
    return G0, ref_points

def naive_kn (X,D,**kwargs):
    '''
    Junta los resultados de naive en una unica clasificacion:
    Agrupa en un subconjunto a los k puntos mas cercanos a un punto de referencia.
    Luego, toma un punto sin clasificacion como referencia y crea el siguiente subconjunto.
    Termina cuando todos los puntos han sido clasificados.

    **kwargs: 
        - k_n (int): numero de vecinos cercanos a considerar. default=50

    Nota: no se garantiza un numero de grupos. maximo numero de grupos = numero de puntos.
    '''
    k_n=kwargs['k_n'] if 'k_n' in kwargs else 50

    # get indixes that sort the distance matrix
    G=np.argsort(D,axis=1)

    # start with the first point
    x_ref=0

    # variables to store results
    ref_points=[]
    G0=np.zeros(shape=(len(X),len(X))) 

    continue_kn=True; i=0
    while continue_kn:
        # store current reference point
        ref_points.append(x_ref)

        # mark all neighbors to x_ref
        mask=G[x_ref][:k_n]
        G0[mask,i]=1

        # find which point has not been assigned to any group and is closest to x_ref
        # make it the next reference point
        continue_kn=False
        for x_ref in G[x_ref][k_n:]: # explore if other close neighbors have not been assigned
            if G0[x_ref].sum()==0: 
                continue_kn=True; break
        i+=1

    return G0[:,:len(ref_points)], ref_points # drop unnecesary columns
