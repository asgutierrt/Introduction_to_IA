# for interactive plots
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_clusters (X,G0,plot_dims,annotations=[],cluster_name='',html_name=''):
  # plot all subsets on G0 on the same figure
  fig = go.Figure()

  # especial formating in case of subset overlap
  subset_overlap = sum(G0.sum(axis=1)!=1)!=0
  marker_symbol='circle-open' if subset_overlap else 'circle' 
  marker_size = lambda i: i+5 if subset_overlap else 5

  n_subsets=G0.shape[1]
  for i in range(n_subsets):
    subset_filter=G0[:,i].astype(bool)
    fig.add_scatter3d(x=X[subset_filter,plot_dims[0]], y=X[subset_filter,plot_dims[1]], 
                      z=X[subset_filter,plot_dims[2]],
                      mode='markers', marker_symbol=marker_symbol, marker_size=marker_size(i),
                      marker=dict(color=[i+1]*sum(subset_filter),colorscale='Viridis', 
                                  opacity=1, cmin=0.8, cmax=n_subsets),
                      name='data points')

  # annotate points
  if annotations!=[]:
    fig.add_scatter3d(x=X[annotations,plot_dims[0]], y=X[annotations,plot_dims[1]], 
                      z=X[annotations,plot_dims[2]], mode='markers+text', 
                      marker=dict(color='red'), marker_size=4, 
                      text=['x[%i]'%n for n in annotations], name='annotations')

  # tight layout
  titulo='clasificacion %s'%cluster_name
  fig.update_layout(title_text=titulo, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
  if html_name!='': fig.write_html(html_name)
  return fig

def subplots(X,G0_list,plot_dims,annotations_list,cluster_name,html_name=''):
  n_subplots = len(G0_list)
  fig = make_subplots(rows=1, cols=n_subplots, 
                      specs=[[{'type': 'surface'}]*n_subplots],
                      subplot_titles=(range(1,n_subplots+1)))

  for fig_ix in range(n_subplots):
    sub_fig = plot_clusters(X,G0_list[fig_ix],plot_dims,annotations_list[fig_ix],cluster_name)
    for trace in sub_fig.data:
      fig.add_trace(trace, row=1, col=fig_ix+1)

  # tight layout
  fig.for_each_annotation(lambda a: a.update(text = "x[%s] as reference"%annotations_list[int(a.text)-1][0]))
  titulo='%s: crear todas las cajas desde un unico punto de referencia'%cluster_name
  fig.update_layout(autosize=True, title_text=titulo, showlegend=False)
  if html_name!='': fig.write_html(html_name)
  return fig