import pandas as pd
import numpy as np
import seaborn as sns

def plot_clusters(X,G,ref_points,grid=None):    
    df_plot=df_plot_PairGrid(X,G,ref_points,grid=grid)
    #df_plot['group']=df_plot['group'].astype(int)
    
    g=sns.PairGrid(df_plot,vars=range(X.shape[1]), hue='group')
    g.map_offdiag(sns.scatterplot,palette='tab10',
                hue=df_plot['group'],style=df_plot['type'],markers={"reference": "P","point": "$\circ$","missed-point":'X'},
                size=df_plot['group'],sizes=(30, 50),alpha=0.9)

    if grid is None:
      g.map_diag(sns.kdeplot, palette='tab10', alpha=0.9)
    
    # format legend
    g.add_legend()
    return g.figure

def df_plot_PairGrid(X,G,ref_points,grid=None):
    df_X=pd.DataFrame(np.hstack([X,np.ceil(G)*range(1,G.shape[1]+1)]))
    mask=G.sum(axis=1)==0

    # classified points
    df_melted_Class=df_X[~mask].melt(id_vars=range(X.shape[1]), value_name='group')
    df_melted_Class=df_melted_Class.drop('variable',axis=1)[df_melted_Class['group']!=0].sort_values('group')
    df_melted_Class['type']='point'

    # not classified points
    df_melted_notClass=df_X[mask].melt(id_vars=range(X.shape[1]), value_name='group')
    df_melted_notClass=df_melted_notClass.drop('variable',axis=1).sort_values('group')
    df_melted_notClass[['group','type']]='not classified','missed-point'
    
    # reference points 
    df_melted_ref=pd.DataFrame(ref_points)
    df_melted_ref['group']=range(1,df_melted_ref.shape[0]+1)
    df_melted_ref['type']='reference'

    if grid is not None:
      # grid points
      df_grid=pd.DataFrame(grid)
      df_grid[['group','type']]='grid','point'
      return pd.concat([df_melted_Class,df_melted_notClass,df_melted_ref,df_grid])

    return pd.concat([df_melted_Class,df_melted_notClass,df_melted_ref])