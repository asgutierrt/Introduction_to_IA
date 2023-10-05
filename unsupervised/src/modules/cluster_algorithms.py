import numpy as np
from modules.support_functions import *
from os.path import join
from modules.visualizations import plot_clusters, subplots

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

class ClusterProblem:
    def __init__(self, X_name, normas, results_path, figs_path, plot_dims):
        # paths to save results and plots
        self.result_path = lambda cluster_func_name: join(results_path, cluster_func_name+'_'+X_name+'.txt')
        self.fig_path = lambda cluster_func_name: join(figs_path, cluster_func_name+'_'+X_name+'.html')

        # data to plot: subtitles (with norm identifier) and dimensions
        self.plot_dims = plot_dims
        self.normas = normas
        self.add_norm_name_to = lambda name: [name+'_'+norma for norma in normas]

    def do_cluster(self, D, cluster_func, **kwargs):
        '''Iterate over the distance matrix and apply the clustering algorithm to each row'''
        G, G_ref_points = zip(*[cluster_func(sub_D,**kwargs) for sub_D in D])
        return G, G_ref_points

    def do_save_clusters(self,G,cluster_func_name):
        save_results(G,self.normas,self.result_path(cluster_func_name))

    def do_plot_clusters(self, X, G, G_ref_points, cluster_func_name, individual_results=False):
        if individual_results:
            for G0, ref_points, cluster_name in zip(G, G_ref_points, self.add_norm_name_to(cluster_func_name)):
                plot_clusters(
                    X=X, G0=G0, 
                    plot_dims=self.plot_dims, annotations=ref_points, 
                    title='classification using '+cluster_name, 
                    html_name=self.fig_path(cluster_name),
                    )
        else:
            subplots(
                X=X,G0_list=G,
                plot_dims=self.plot_dims,annotations_list=G_ref_points,
                subtitles=self.add_norm_name_to('distance'),
                title='classificacion using %s'%cluster_func_name,
                html_name=self.fig_path(cluster_func_name),
                )
    
    def do_ClusterPipeline(self, X, D, cluster_func, cluster_func_name, **kwargs):
        G, G_ref_points = self.do_cluster(D, cluster_func, **kwargs)
        self.do_save_clusters(G, cluster_func_name)
        if len(self.plot_dims)>2:
            self.do_plot_clusters(X, G, G_ref_points, cluster_func_name, individual_results=False)
        return G, G_ref_points


'''
tried to parallelize the clustering algorithms - do not run :(
info here: https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
and here: https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
and here: https://medium.com/@deveshparmar248/python-multiprocessing-maximize-the-cpu-utilization-eec3b60e6d40
import multiprocessing
# apply_cluster_func= lambda cluster_func, arr, kwargs: cluster_func(D=arr,**kwargs)
def parallel_apply_along_axis(cluster_func, arr, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    # Chunks for the mapping (only a few chunks):
    chunks = [(cluster_func, sub_arr, kwargs) for sub_arr in arr]
    pool = multiprocessing.Pool(3) #multiprocessing.cpu_count()
    individual_results = pool.map(func=apply_cluster_func, iterable=chunks)
    # Freeing the workers:
    pool.close()
    pool.join()
    return np.concatenate(individual_results)
'''


def naive (D,**kwargs):
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

def naive_boxes (D,**kwargs):
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
                                    n_groups=n_groups).reshape((D.shape[1],-1))
    
    x_ref=0 # el primer subconjunto se forma con los puntos cerca a x_ref
    ref_points=[] # almacenar que punto de referencia han sido usados
    G0=np.zeros(shape=(D.shape[1],n_groups)) # almacenar la clasificacion de cada punto

    for i in range(n_groups):
        # store current reference point
        ref_points.append(x_ref)

        # mark all points that are a threshold distance to x_ref
        mask=G_threshold[x_ref]<1
        G0[mask,i]=1

        # find the closest point to x_ref that is not yet classified
        # make it the next reference point
        x_ref=unclassified_points(G_threshold[x_ref],G0).argmin()
    
    return (G0, ref_points)

def naive_kn (D,**kwargs):
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
    G0=np.zeros(shape=(D.shape[1],D.shape[1])) 

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

def density_substraction(D,ra,kind='substractive',**kwargs):
    
    # define parameters
    if kind=='substractive':
        D_intra=D_entre=D
    if kind=='mountain':
        D_intra=D[0]; D_entre=D[1]
    
    stop_criteria=kwargs['stop_criteria'] if 'stop_criteria' in kwargs else 'k_centroids'
    rb=kwargs['rb'] if 'rb' in kwargs else ra*1.15 
    radius=lambda r: 4/r**2

    # 1. initialize densities
    if kind=='mountain':
        # define potential cluster centers based on density over balls of radius ra
        X_density=np.apply_along_axis(lambda x: np.exp(-radius(ra) * x), 0, D_intra).sum(axis=1)
        
    if kind=='substractive':
        # define potential cluster centers based on density over balls of radius ra
        X_density=np.zeros(D_intra.shape[0])

        # evaluate density around each data point (not grid points)
        for i in range(D_intra.shape[0]):
            for j in range(i + 1, D_intra.shape[0]): # make use of simmetry of the distance matrix
                value = np.exp(-radius(ra) * D_intra[i,j])
                # update density
                X_density[i] += value; X_density[j] += value
    
    # 2. find cluster centers
    ref_points=[]
    
    while True:
        # find the best centroid (the one with the highest density)
        best_centroid_ix=X_density.argmax()
        centroid_density=X_density[best_centroid_ix]
        
        # reduce density values on a radius rb around chosen centroid
        for i in range(X_density.shape[0]):
            value = np.exp(-radius(rb) * D_entre[best_centroid_ix,i])
            # update density
            X_density[i] -= centroid_density*value
        
        # update stop criteria
        threshold=kwargs['epsilon']*centroid_density if stop_criteria=='low_density' and len(ref_points)==0 else 0
        stop=update_stop_criteria(stop_criteria, ref_points=ref_points, threshold=threshold, centroid_density=centroid_density, **kwargs)
        if stop: break
        ref_points.append(best_centroid_ix)


    # 3. make groups from the reference points: fuzzy rule based on density evaluation
    G=np.zeros(shape=(D_intra.shape[1],len(ref_points)))
    for group, centroid_ix in enumerate(ref_points):
        G[:,group]=np.exp(-radius(ra) * D_intra[centroid_ix,:])
    G/=G.sum(axis=1,keepdims=True)

    ## define threshold to define if a point belongs to a group or not
    mask=G>G.mean(axis=1,keepdims=True)
    G = G*mask
    return G, ref_points

# define stop criteria for density substraction
def update_stop_criteria(x,**kwargs):
    if x=='k_centroids':
        return len(kwargs['ref_points'])==kwargs['k']
    if x=='low_density':
        return kwargs['centroid_density']<kwargs['threshold']
    return False