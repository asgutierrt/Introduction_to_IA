import numpy as np

def threshold_distance(x,n_groups):
    # threshold defined by number of groups
    return x/((x.max()+10**-10-x.min())/n_groups)

def naive (D,n_groups=3,norm_ix=0):
    # clasificar segun la distancia threshold a cada punto
    # calcular la distancia threshold con la minima y maxima distancia a cada punto
    G=np.apply_along_axis(threshold_distance, 1, D[norm_ix], n_groups=n_groups).astype(int)
    return G

def unclassified_points(points,G0):
    return np.ma.masked_array(points, (G0.sum(axis=1)!=0))

def naive_boxes (X,D,n_groups=3,norm_ix=0):
    # clasifica segun una distancia threshold calculada con la minima y maxima distancia entre puntos en el dataset
    G_threshold=np.apply_along_axis(threshold_distance, 0, D[norm_ix].flatten(),
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

def naive_kn (X,D,k_n=50,norm_ix=0):
    # sorted indixes
    G=np.argsort(D[norm_ix],axis=1)

    x_ref=0 # el primer subconjunto se forma con los puntos cerca a x_ref
    ref_points=[] # almacenar que punto de referencia han sido usados
    # group assignment matrix: theres is a maximum os subsets equal to the number of points
    G0=np.zeros(shape=(len(X),len(X))) 

    end_kn=False; i=0
    while not bool(end_kn):
        # store current reference point
        ref_points.append(x_ref)

        # mark all neighbors to x_ref
        mask=G[x_ref][:k_n]
        G0[mask,i]=1

        # find which point has not been assigned to any group and is closest to x_ref
        # make it the next reference point
        for x_ref in G[x_ref][k_n:]: # explore if other close neighbors have not been assigned
            end_kn=True
            if G0[x_ref].sum()==0: 
                end_kn=False; break
        i+=1

    return G0[:,:len(ref_points)], ref_points # drop unnecesary columns
