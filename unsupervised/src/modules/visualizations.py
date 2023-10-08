import pandas as pd
import numpy as np
import seaborn as sns

def plot_clusters(i,X,G,G_ref_points,grid=None):
    df_plot=df_plot_PairGrid(i,X,G,G_ref_points,grid=grid)
    g=sns.PairGrid(df_plot,vars=range(X.shape[1]))
    g.map_offdiag(sns.scatterplot,palette='tab10',
                hue=df_plot['group'],style=df_plot['type'],markers={"reference": "X","point": "$\circ$"},
                size=df_plot['group'],sizes=(30, 50),alpha=0.5)
    
    if grid is None:
      g.map_diag(sns.kdeplot,hue=df_plot['group'])

    # format legend
    g.add_legend()
    return g.figure

def df_plot_PairGrid(i,X,G,G_ref_points,grid=None):
    df_X=pd.DataFrame(np.hstack([X,np.ceil(G[i])*range(1,G[i].shape[1]+1)]))
    mask=G[i].sum(axis=1)==0

    # classified points
    df_melted_Class=df_X[~mask].melt(id_vars=range(X.shape[1]), value_name='group')
    df_melted_Class=df_melted_Class.drop('variable',axis=1)[df_melted_Class['group']!=0].sort_values('group')
    df_melted_Class['type']='point'

    # not classified points
    df_melted_notClass=df_X[mask].melt(id_vars=range(X.shape[1]), value_name='group')
    df_melted_notClass=df_melted_notClass.drop('variable',axis=1).sort_values('group')
    df_melted_notClass[['group','type']]='not classified','point'
    
    if grid is not None:
      # grid points
      df_grid=pd.DataFrame(grid)
      df_grid[['group','type']]='grid','point'
      
      # reference points in grid
      df_melted_ref=df_grid.loc[G_ref_points[i]]
      df_grid.drop(index=G_ref_points[i],inplace=True)
      df_melted_ref['group']=range(1,df_melted_ref.shape[0]+1)
      df_melted_ref['type']='reference'
      
      return pd.concat([df_melted_Class,df_melted_notClass,df_melted_ref,df_grid])
    else:
      # reference points in X
      df_melted_ref=df_X.loc[G_ref_points[i],range(X.shape[1])]
      df_melted_ref['group']=range(1,df_melted_ref.shape[0]+1)
      df_melted_ref['type']='reference'
      
      return pd.concat([df_melted_Class,df_melted_notClass,df_melted_ref])