# for interactive plots
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_clusters (X,G0,plot_dims,annotations=[],title='',html_name=''):
  # plot all subsets on G0 on the same figure
  fig = go.Figure()

  # especial formating in case groups overlap 
  subset_overlap = sum(G0.sum(axis=1)!=1)!=0
  marker_symbol='circle-open' if subset_overlap else 'circle' 
  marker_size = lambda i: i+5 if subset_overlap else 5

  n_subsets=G0.shape[1] # number of groups in G0
  for i in range(n_subsets):
    # filter points in group i
    subset_filter=G0[:,i].astype(bool) 

    fig.add_scatter3d(
      # data points
      x=X[subset_filter,plot_dims[0]], 
      y=X[subset_filter,plot_dims[1]], 
      z=X[subset_filter,plot_dims[2]],
      # markers
      mode='markers', marker_symbol=marker_symbol, marker_size=marker_size(i),
      # color by group: share colorbar across groups
      marker=dict(color=[i+1]*sum(subset_filter),
                  colorscale='Viridis', 
                  opacity=1, cmin=0.8, cmax=n_subsets),
    )

  # annotate points 
  if annotations!=[]:
    fig.add_scatter3d(
      # data points
      x=X[annotations,plot_dims[0]], 
      y=X[annotations,plot_dims[1]], 
      z=X[annotations,plot_dims[2]], 
      # annotations
      mode='markers+text', marker=dict(color='red'), marker_size=4, 
      text=['x[%i]'%n for n in annotations],
    )

  # add title, and set tight layout
  fig.update_layout(title_text=title, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
  # save figure
  if html_name!='': fig.write_html(html_name)
  return fig

def subplots(X,G0_list,plot_dims,annotations_list,subtitles,title,html_name=''):
  n_subplots = len(G0_list)
  max_cols=3
  nrows=1+(n_subplots-1)//max_cols; ncols=min(n_subplots, max_cols)
  fig = make_subplots(rows=nrows, cols=ncols, 
                      specs=[[{'type': 'surface'}]*ncols]*nrows,
                      subplot_titles=subtitles)

  for fig_ix in range(n_subplots):
    sub_fig = plot_clusters(X,G0_list[fig_ix],plot_dims,annotations_list[fig_ix])
    for trace in sub_fig.data:
      fig.add_trace(trace, row=1+int(fig_ix/max_cols), col=1+fig_ix%max_cols)

  # tight layout
  fig.update_layout(autosize=True, title_text=title, showlegend=False)
  if html_name!='': fig.write_html(html_name)
  return fig