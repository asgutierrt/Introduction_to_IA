# for interactive plots
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


def do_plot_clusters(self, X, grid, G, G_ref_points, cluster_func_name, individual_plots=False, verbose=False):
        if individual_plots:
            for G0, ref_points, cluster_name in zip(G, G_ref_points, self.add_norm_name_to(cluster_func_name)):
                fig=plot_clusters(
                        X=X, G0=G0, grid=grid,
                        plot_dims=self.plot_dims, annotations=ref_points, 
                        title='classification using '+cluster_name, 
                        html_name=self.fig_path(cluster_name),
                        )
                if verbose: fig.show()
        else:
            fig=subplots(
                    X=X,grid=grid,G0_list=G,
                    plot_dims=self.plot_dims,annotations_list=G_ref_points,
                    subtitles=self.add_norm_name_to('distance'),
                    title='classificacion using %s'%cluster_func_name,
                    html_name=self.fig_path(cluster_func_name),
                    )
        if verbose: fig.show()


def plot_clusters_aux (fig,X,G0,plot_dims,annotations,
                       unclassified_color='gray',standard_marker_size=5,
                       grid=[]):
  
  n_groups=G0.shape[1]
  # number of groups each point belongs to
  x_color_filter = np.count_nonzero(G0,axis=1) 
  x_unlabelled = x_color_filter==0 # points not in any group
  x_multi_labelled = x_color_filter>1 # points in more than one group

  # format the drawing of X points: especiall format for multi-labelled points
  marker_size=lambda i: np.array([standard_marker_size+2*i if x else standard_marker_size for x in x_multi_labelled])
  marker_symbol=np.array(['x' if x else 'circle' for x in x_multi_labelled])

  fig_format=dict(
    mode='markers',
    marker=dict(colorscale='Viridis', cmin=0.8, cmax=n_groups, opacity=1))

  # draw points in each group
  for i in range(n_groups):
    # data points
    group_filter=G0[:,i].astype(bool)
    data_points=[X[group_filter,x] for x in plot_dims]
    data_points=dict(x=data_points[0], y=data_points[1], z=data_points[2])

    # update marker color
    fig_format['marker']['color']=[i+1]*sum(group_filter)
    
    # add points to figure
    fig.add_scatter3d(**data_points, 
                      marker_symbol=marker_symbol[group_filter], 
                      marker_size=marker_size(i)[group_filter],
                      **fig_format)

  # draw not classified points
  if x_unlabelled.sum()>0:
    # data points
    data_points=[X[x_unlabelled,x] for x in plot_dims]
    data_points=dict(x=data_points[0], y=data_points[1], z=data_points[2])
    
    # update marker color
    fig_format['marker']['color']=[unclassified_color]*sum(group_filter)

    # add points to figure
    fig.add_scatter3d(**data_points, 
                      marker_symbol=marker_symbol[x_unlabelled], 
                      marker_size=marker_size(-1)[x_unlabelled],
                      **fig_format)
  
  # draw points in the grid if available
  if grid is not None: 
    # data points
    data_points=[grid[:,x] for x in plot_dims]
    data_points=dict(x=data_points[0], y=data_points[1], z=data_points[2])

    # update marker color
    fig_format['marker']['color']=[unclassified_color]*len(grid)

    # add points to figure
    fig.add_scatter3d(**data_points, 
                      marker_symbol=['square']*len(grid),
                      marker_size=[standard_marker_size]*len(grid),
                      **fig_format)

  # draw annotations
  if annotations!=[]:
    # data points
    if grid!=[]: # draw points in the grid if available
      data_points=[grid[annotations,x] for x in plot_dims]
    else: 
      data_points=[X[annotations,x] for x in plot_dims]
    data_points=dict(x=data_points[0], y=data_points[1], z=data_points[2])

    # add points to figure
    fig_format.update({'mode':'markers+text'})
    fig_format['marker']['color']=list(range(1,len(annotations)+1))
    fig.add_scatter3d(**data_points,
                      marker_symbol=['diamond']*len(annotations),
                      marker_size=[standard_marker_size+3]*len(annotations),
                      text=['x[%i]'%n for n in annotations],
                      **fig_format)

def plot_clusters(X,G0,plot_dims,annotations=[],grid=None,title='',html_name=''):
  # plot all sets of points on the same figure
  fig = go.Figure()  
  plot_clusters_aux (fig,X,G0,plot_dims,annotations,grid=grid)
  # add title, and set tight layout
  fig.update_layout(title_text=title, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
  # save figure
  if html_name!='': fig.write_html(html_name)
  return fig

def subplots(X,grid,G0_list,plot_dims,annotations_list,subtitles,title,html_name=''):
  n_subplots = len(G0_list)
  max_cols=3
  nrows=1+(n_subplots-1)//max_cols; ncols=min(n_subplots, max_cols)
  fig = make_subplots(rows=nrows, cols=ncols, 
                      specs=[[{'type': 'surface'}]*ncols]*nrows,
                      subplot_titles=subtitles)

  for fig_ix in range(n_subplots):
    sub_fig = plot_clusters(X,grid,G0_list[fig_ix],plot_dims,annotations_list[fig_ix])
    for trace in sub_fig.data:
      fig.add_trace(trace, row=1+int(fig_ix/max_cols), col=1+fig_ix%max_cols)

  # tight layout
  fig.update_layout(autosize=True, title_text=title, showlegend=False)
  if html_name!='': fig.write_html(html_name)
  return fig