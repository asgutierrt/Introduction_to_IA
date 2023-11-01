from modules.cluster_algorithms import *
from modules.cluster_object import ClusterProblem

from os.path import join, dirname
from os import chdir


def main(filename='iris'):
    # 1. set working directory and paths
    project_wd=dirname(dirname(__file__)); chdir(project_wd)
    figs_path= join(project_wd,'reports','figures')
    results_path= join(project_wd,'reports','group_matrix')

    # 2. load and process data
    normas=['euclidea','mahalanobis','coseno','manhattan','Lp=3']
    pipe=ClusterProblem(results_path,figs_path,filename,x_treatment='original',normas=normas)

    # 1. naive clustering: cajitas
    G, G_ref_points = pipe.do_cluster(naive_boxes, n_groups=2)

    # 2. naive clustering: k-vecinos
    G, G_ref_points = pipe.do_cluster(naive_kn, k_n=70)
    
    # 3. mountain clustering (with grid data)
    G, G_ref_points = pipe.do_cluster(density_substraction, ra=0.3, kind='mountain', stop_criteria='unique_centroids', only_centroids=False)

    # 4. substractive clustering
    G, G_ref_points = pipe.do_cluster(density_substraction, ra=0.3, kind='substractive', stop_criteria='unique_centroids', only_centroids=False)

    #5. Hard k-means clustering
    G, G_ref_points = pipe.do_cluster(kn_HardCluster, k_n=5, tol=1e-3, norma='euclidea')
    
    #6. Fuzzy k-means clustering
    G, G_ref_points = pipe.do_cluster(kn_FuzzyCluster, k_n=5, tol=1e-3, m=2, norma='euclidea')

    print('clustering algorithms finished')
    
if __name__=="__main__":
    # main(filename='water_potability.txt')
    main() #iris 