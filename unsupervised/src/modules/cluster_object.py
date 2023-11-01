import matplotlib.pyplot as plt
import seaborn as sns
from modules.indices import internal_indices
import pandas as pd
from itertools import product
from os.path import join
import os
import numpy as np

import umap
from modules.load_data import load_data
from modules.autoencoder import encode
from modules.distances import get_distance_matrix, covarianza_inversa
from modules.cluster_algorithms import naive_boxes, density_substraction, naive_kn, kn_HardCluster, kn_FuzzyCluster
from modules.support_functions import make_grid, encode_array, defuzzyfy


class ClusterProblem:
    def __init__(self, results_path, figs_path, filename, x_treatment='original', normas=['euclidea']):
        # setup to load data
        self.load_data (filename, x_treatment)
        self.normas = normas
        self.add_norm_name_to = lambda name: [name+'_'+norma for norma in normas]

        # paths to save results and plots
        self.result_path = lambda cluster_func_name: join(results_path, cluster_func_name+'_'+x_treatment+'.txt')
        self.fig_path = lambda cluster_func_name: join(figs_path, cluster_func_name+'_'+x_treatment+'.png')

        # data to work with
        self.cov_i = covarianza_inversa(self.X)
        self.D_XToX = get_distance_matrix(self.X,self.X,self.cov_i,norms=self.normas)

        # grid to work with
        self.do_grid(n_intervals=None)

    
    def load_data (self, filename, x_treatment):
        self.X = load_data(os.path.join('..','data',filename),verbose=True)
        if x_treatment=='encoded':
            self.X = encode(self.X, encoding_layers_dims=[int(self.X.shape[1]*1.2),int(self.X.shape[1]*1.5)], verbose=True)
        elif x_treatment=='umap':
            print('calculating umap embedding')
            self.X = umap.UMAP().fit_transform(self.X)

    def do_grid(self,n_intervals=None):
        '''Create a grid of n_intervals in each dimension'''
        self.grid = make_grid(self.X,n_intervals=n_intervals)
        self.D_GridToX = get_distance_matrix(self.grid,self.X,self.cov_i,norms=self.normas)
        self.D_GridToGrid = get_distance_matrix(self.grid,self.grid,covarianza_inversa(self.grid),norms=self.normas)

    def plot_distances(self,D):
        """
        Creates html visualizations of the distance matrix calculated with different norms.
        """
        nrows=int(np.ceil(len(self.normas)/2)); ncols=2
        fig=plt.figure(figsize=(9,7))
        for i, norma in enumerate(self.normas):
            ax=plt.subplot(nrows,ncols,i+1)
            cax=ax.matshow(D[i], cmap=plt.cm.Blues.reversed(), aspect='auto')
            ax.set_title(norma)
            fig.colorbar(cax, ax=ax,fraction=0.046)
        plt.tight_layout()
        fig.savefig(join(self.fig_path('distances')))

    def do_cluster(self, cluster_func, **kwargs):
        '''Iterate over the distance matrix and apply the clustering algorithm to each row'''
        # iterate over arguments that are lists or arrays except X
        kwargs.update({'X':self.X, 'cov_i':self.cov_i, 'normas':self.normas, 'D':self.D_XToX, 
                       'D_intra':self.D_GridToX, 'D_entre':self.D_GridToGrid})
        iter_kwargs = {arg_name: arg_vals for arg_name, arg_vals in kwargs.items() if isinstance(arg_vals,(list,np.ndarray)) 
                       and arg_name!='X' and arg_name!='cov_i'}
        results=[]
        for iter_args in [dict(zip(iter_kwargs.keys(), v)) for v in zip(*iter_kwargs.values())]:
            kwargs.update(iter_args)
            results.append(cluster_func(**kwargs))
        return zip(*results)
    
    def centroids_by_density (self, density_kind, ra, norma_i):
        _, centroids_ix = density_substraction(X=self.X,
                                               D=self.D_XToX[norma_i], D_intra=self.D_GridToX[norma_i], D_entre=self.D_GridToGrid[norma_i], 
                                               ra=ra, kind=density_kind, stop_criteria='unique_centroids', 
                                               only_centroids=True)
        if density_kind=='substractive':    
            centroids=self.X[centroids_ix]
        elif density_kind=='mountain':
            centroids=self.grid[centroids_ix]
        return centroids
    
    def cluster_by_neighbors (self, cluster_kind, n_groups, norma_i, centroids=[], m=2):
        if cluster_kind=='hard':
            U, ref_points = kn_HardCluster(norma=self.normas[norma_i], X=self.X, cov_i=self.cov_i, 
                                           k_n=n_groups, centroids=centroids)
            G=self.do_format_pertenence_matrix(U, ref_points)
            return self.drop_empty_clusters(G, ref_points)
        
        elif cluster_kind=='fuzzy':
            U, ref_points = kn_FuzzyCluster(X=self.X, cov_i=self.cov_i, k_n=n_groups, norma=self.normas[norma_i], m=m)
            # make hard clustering
            G=self.do_format_pertenence_matrix(U, ref_points)
            return self.drop_empty_clusters(G, ref_points)
    
    def do_format_pertenence_matrix(self, U, ref_points):
        if U.ndim==1: # encode as matrix
            G=np.zeros((U.shape[0],ref_points.shape[0]))
            for x_ix, x_group in enumerate(U): G[x_ix,x_group]=1
            return G
        else: # make hard clustering (not fuzzy)
            G=np.zeros_like(U)
            for x_ix, x_group in enumerate(U.argmax(axis=1)): G[x_ix,x_group]=1
            return G
        
    def drop_empty_clusters(self, G, ref_points):
        drop_empty_mask=G.sum(axis=0)!=0
        return G[:,drop_empty_mask], ref_points[drop_empty_mask]
    
    def single_pipeline(self,density_kind, ra, norma_i, m=2, previous_k=[]):
        centroids = self.centroids_by_density(density_kind, ra, norma_i)
        k=centroids.shape[0]
        if k in previous_k: raise ValueError('k=%d already calculated'%k)
        # hard clustering
        G, ref_points = self.cluster_by_neighbors('hard', k, norma_i)
        db, dn, ch = internal_indices(self.X, G, self.normas[norma_i], self.cov_i, ref_points)
        yield (G, ref_points), ref_points.shape[0], db, dn, ch  
        # fuzzy clustering
        G, ref_points = self.cluster_by_neighbors('fuzzy', k, norma_i, m=m)
        db, dn, ch = internal_indices(self.X, G, self.normas[norma_i], self.cov_i, ref_points)
        yield (G, ref_points), ref_points.shape[0], db, dn, ch 
        # keep track of number of clusters already explored
        yield previous_k+[k]
    
    def create_results_df(self,density_algs,ra_percentile=30,ra_range=[1,1.1,0.9]):
        ra_base=[(norma, np.percentile(self.D_XToX[norma_i], ra_percentile)) for norma_i, norma in enumerate(self.normas)]
        ra_values=dict([(i, [ra_*x for x in ra_range]) for i, ra_ in ra_base])

        indexes=((norma,density_alg,ra) for norma in self.normas for density_alg in density_algs for ra in ra_values[norma])
        columns=pd.MultiIndex.from_product([['k_n','c_n'],['matrix results','n_clusters','davies bouldin', 'dunn', 'calinski-harabasz']])
        self.results=pd.DataFrame(index=indexes, columns=columns)
        self.results.index.name=('metric','density algorithm','ra')
        return ra_values

    def fill_results(self,name_ix, name_cols,info):
        self.results.loc[name_ix,(name_cols,'matrix results')]=[info[0]]
        self.results.loc[name_ix,(name_cols,['n_clusters','davies bouldin', 'dunn', 'calinski-harabasz'])]=info[1:]

    def do_ClusterPipeline(self, density_algs=['substractive','mountain'], ra_range=[0.3,0.5,1,2,3,4,6,10]):
        ra_values=self.create_results_df(density_algs,ra_percentile=40,ra_range=ra_range)
        for norma_i, norma in enumerate(self.normas):
            print('calculating for norm %s'%norma)
            k_history=[]
            for kind in density_algs:
                for ra in ra_values[norma]:
                    try:
                        kn, cn, k_history = self.single_pipeline(kind, ra, norma_i, previous_k=k_history)
                        self.fill_results([(norma,kind,ra)], 'k_n', kn)
                        self.fill_results([(norma,kind,ra)], 'c_n', cn)
                    except ValueError as e:
                        #print(e)
                        continue
                    except Exception as e:
                        print('error in %s %s %s: %s'%(norma,kind,ra,e))
        self.results.dropna(inplace=True,how='all')
        self.results.index=pd.MultiIndex.from_tuples(self.results.index, names=['metric','density algorithm','ra'])

    def plot_results(self):
        metric_labels, labels=[], []
        for metric,alg,ra in self.results.index:
            # first_val=round(ra,2)
            first_val=self.results.loc[(metric,alg,ra),('k_n','n_clusters')]
            if metric not in metric_labels: 
                labels.append(str(first_val)+'\n'+alg+'\n'+metric)
                metric_labels.append(metric); alg_labels=[alg]
            elif alg not in alg_labels: 
                labels.append(str(first_val)+'\n'+alg)
                alg_labels.append(alg) 
            else:
                labels.append(first_val)

        sns.set()
        fig,axs=plt.subplots(3,1,figsize=(25,5),sharex=True)

        plot_indices=['davies bouldin','dunn','calinski-harabasz']
        results_plot=self.results.loc[:,('k_n',plot_indices)]
        axs=results_plot.plot(subplots=True,legend=False,ax=axs)
        self.results.loc[:,('c_n',plot_indices)].plot(subplots=True,legend=False,ax=axs,style='--')

        # set indices titles
        for i, ax in enumerate(axs): ax.set_title(plot_indices[i]+' index',loc='right')

        # set x labels (formated)
        label_ax=axs[2]
        label_ax.set_xticks(range(len(labels)))
        label_ax.set_xticklabels(labels, ha='center', size=9)
        label_ax.set_xlabel('k\nalgorithm\nmetric',ha='left',labelpad=10)
        
        order_ascending={'davies bouldin':False,'dunn':True,'calinski-harabasz':True} # order of the indices
        color_by={'Worst':'red','Best':'green'}
        ind_list=[]
        for i, indice in enumerate(order_ascending.keys()):
            for mark_fig in ['Worst','Best']:
                inverted_order=lambda x: not x if mark_fig=='Worst' else x
                sub_results=self.results.xs(indice,level=1,axis=1).to_numpy()
                best_indice = sub_results.argmax() if inverted_order(order_ascending[indice]) else sub_results.argmin()
                ind = np.unravel_index(best_indice, sub_results.shape)

                fig.axes[i].scatter(ind[0],sub_results[ind],color=color_by[mark_fig],marker='o')
                ind_list.append(ind)
        yield ind_list

        fig.legend(['k_n','c_n','worst','best'], loc='upper right', bbox_to_anchor=(0.2, 0.999),ncols=2)
        fig.suptitle('Clustering Results per Intra-Cluster Index')
        fig.savefig(join(self.fig_path('indices')))
        yield fig