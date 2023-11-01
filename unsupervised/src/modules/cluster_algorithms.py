from modules.support_functions import *
from modules.visualizations import plot_clusters
from modules.distances import get_distance_matrix
from os.path import join
import numpy as np

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
    def __init__(self, X_name, normas, results_path, figs_path):
        # paths to save results and plots
        self.result_path = lambda cluster_func_name: join(results_path, cluster_func_name+'_'+X_name+'.txt')
        self.fig_path = lambda cluster_func_name: join(figs_path, cluster_func_name+'_'+X_name+'.html')

        # data to plot: subtitles (with norm identifier) and dimensions
        self.normas = normas
        self.add_norm_name_to = lambda name: [name+'_'+norma for norma in normas]

    def do_cluster(self, cluster_func, **kwargs):
        '''Iterate over the distance matrix and apply the clustering algorithm to each row'''
        D=kwargs.pop('D') if 'D' in kwargs else self.normas
        G, G_ref_points = zip(*[cluster_func(D[i],**kwargs) for i in range(len(self.normas))])
        return G, G_ref_points

    def do_save_clusters(self,G,cluster_func_name):
        save_results(G,self.normas,self.result_path(cluster_func_name))

    def do_eval_clusters():
        return 0
    
    def do_ClusterPipeline(self, cluster_func, cluster_func_name, **kwargs):
        G, G_ref_points = self.do_cluster(cluster_func, **kwargs)
        self.do_save_clusters(G, cluster_func_name)
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
    G=np.argsort(D)

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

def density_substraction(X,D,D_intra,D_entre,ra,kind,stop_criteria,**kwargs):
    # define parameters
    if kind=='substractive':
        D_intra=D_entre=D
    
    rb=kwargs['rb'] if 'rb' in kwargs else ra*1.15 
    radius=lambda r: 1/(4*r**2) # radius of the ball

    # 1. initialize densities
    X_density=np.apply_along_axis(lambda x: np.exp(-radius(ra) * x), 0, D_intra).sum(axis=1)

    # 2. find cluster centers
    ref_points=[]
    
    while True:
        # find the best centroid (the one with the highest density)
        best_centroid_ix=np.nanargmax(X_density)
        centroid_density=X_density[best_centroid_ix]
        
        # reduce density values on a radius rb around chosen centroid
        reduction_factor = np.apply_along_axis(lambda x: np.exp(-radius(rb) * x), 0, D_entre[best_centroid_ix])
        X_density -= centroid_density*reduction_factor
        
        # update stop criteria
        threshold=kwargs['epsilon']*centroid_density if stop_criteria=='low_density' and len(ref_points)==0 else 0
        stop=update_stop_criteria(stop_criteria, ref_points=ref_points, threshold=threshold, 
                                  centroid_density=centroid_density, 
                                  best_centroid_ix=best_centroid_ix, **kwargs)
        if stop: break
        ref_points.append(best_centroid_ix)

    G=np.zeros(shape=(D_intra.shape[1],len(ref_points)))
    if kwargs['only_centroids']: 
        return G, ref_points
    else:
        # 3. make groups from the reference points: fuzzy rule based on density evaluation
        for group, centroid_ix in enumerate(ref_points):
            G[:,group]=np.exp(-radius(ra) * D_intra[centroid_ix,:])
        
        point_densities_sum = G.sum(axis=1,keepdims=True)
        G/=np.where(point_densities_sum==0, np.nan, point_densities_sum)
        G=np.nan_to_num(G)

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
    if x=='unique_centroids':
        return kwargs['best_centroid_ix'] in kwargs['ref_points']
    return False

def kn_HardCluster(X,norma,cov_i,k_n,tol=1e-3,**kwargs):
    # initialize centroids and cost function
    centroids=np.random.default_rng().choice(X,k_n,replace=False)

    J=0
    while True:
        # find distances to centroids
        D_CentroidToX=get_distance_matrix(centroids,X,cov_i,norms=[norma]).squeeze()

        # assign points to closest centroid
        U=D_CentroidToX.argmin(axis=0)

        new_J=0
        for ci in range(k_n):
            mask=U==ci
            # update cost function
            new_J+=np.sum(D_CentroidToX[ci][mask]**2)
            # update centroid
            if mask.sum()==0:
                centroids[ci]=np.random.default_rng().choice(X)
            else:
                centroids[ci]=X[mask].mean(axis=0)
        # check convergence
        if np.abs(J-new_J)<tol:
            break
        J=new_J
    return U, centroids

def kn_FuzzyCluster(X,cov_i,k_n,norma,m=1.01,tol=1e-3, **kwargs):
    # fuzziness parameter: m->1 -> hard clustering

    # initialize membership matrix
    U = np.random.rand(X.shape[0],k_n)
    U /= np.sum(U,axis=1,keepdims=True)

    last_J=0
    while True:
        # update centroids
        centroids = ((U.T**m).dot(X))/((U**m).sum(axis=0,keepdims=True).T)
        D_CentroidToX=get_distance_matrix(X,centroids,cov_i,norms=[norma]).squeeze()

        # calculate cost function
        J = np.sum(U**m*D_CentroidToX**2)

        # check convergence
        if np.abs(last_J-J)<tol: 
            break
        last_J=J

        # update membership matrix
        U_update = 1/(D_CentroidToX**(2/(m-1)))
        U_update /= np.sum(U_update,axis=1,keepdims=True)
        U *= U_update

    return U, centroids
