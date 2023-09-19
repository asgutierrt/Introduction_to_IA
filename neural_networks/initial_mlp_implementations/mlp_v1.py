# instalar e importar las librerias necesarias
try:
    import numpy as np
    from sympy import *
    import pandas as pd
    import matplotlib.pyplot as plt
    import tqdm
except:
    import os
    cmds=['pip install --upgrade pip', 'pip install numpy sympy pandas matplotlib tqdm']
    for cmd in cmds: os.system(cmd)

def layer (x,m,phi,l):
  # tamano de la entrada a cada perceptron en la capa
  n=int(x.shape[0])
  # crear la matriz de variables w para la capa
  W=Matrix(symarray('W'+l, (m, n)))
  # calcular el campo local inducido en cada perceptron
  V=W*x
  # funcion de activacion evaluada en el campo de los m perceptrones
  Y=Matrix(m,1, lambda i,j: phi.subs({"v":V[i,j]}))
  return Y, W, V

# descenso por el gradiente
def dJ_dw_func(phi,v,w,init_local_grad,Y_prev,dict_vals):
  # phi' evaluado en los campos de la capa 
  dy_dv=Matrix(w.shape[0],1, lambda i,j: diff(phi,"v").subs({"v":v[i].evalf(subs=dict_vals)}))
  # terminar de calcular el gradiente local
  local_grad=init_local_grad.multiply_elementwise(dy_dv)
  # calcular el gradiente
  dJ_dw=local_grad*Y_prev.evalf(subs=dict_vals).T
  # iniciar el calculo del gradiente local para la capa siguiente
  next_local_grad=w.T*local_grad
  return dJ_dw, local_grad, next_local_grad.evalf(subs=dict_vals)

def update_w_grad(phi,v,w,Y_prev,init_local_grad,X,mx,w_vals,dict_vals,eta):
  dJ_dw, local_grad, next_init_local_grad=dJ_dw_func(phi,v,w,init_local_grad,Y_prev,dict_vals)
  w_grads=dJ_dw; w_vals+=eta*w_grads
  return w_vals, w_grads, local_grad, next_init_local_grad

def MLP_3capas(X,Y_d,eta):
  # modelo MLP
  # funciones de activacion
  a=1; b=0; v = symbols("v")
  phi_1=a*v+b # lineal
  phi_2=(exp(v)-exp(-v))/(exp(v)+exp(-v)) # tanh
  phi_3=1/(1+exp(-v)) # sigmoid

  # Planteamiento forward
  n=X.shape[1]; mx=Matrix(symarray('X', n)) # input
  mi=2; phi_i=phi_2; li, wi, vi=layer(mx,mi,phi_i,"i")
  mj=2; phi_j=phi_3; lj, wj, vj=layer(li,mj,phi_j,"j")
  mk=Y_d.shape[1]; phi_k=phi_1; lk, wk, vk=layer(lj,mk,phi_k,"k")
  model, w_vars= lk, list(wi)+list(wj)+list(wk) # output

  # w inicial aleatoria
  np.random.seed(15)
  wi_vals,wj_vals,wk_vals=[Matrix(np.random.rand(w.shape[0],w.shape[1])) for w in [wi,wj,wk]]
  dict_vals=dict(set(zip(wi,wi_vals))|set(zip(wj,wj_vals))|set(zip(wk,wk_vals)))

  # calcular gradientes para el punto p y, en cada capa, actualizar w de manera secuencial
  epocas=5
  w_vals=[list(wi_vals)+list(wj_vals)+list(wk_vals)]; w_local_grads=[]; w_grads=[]
  for iteracion in range(len(X)*epocas):
    p=iteracion%len(X)
    dict_vals.update(dict(set(zip(mx,X[p]))))
    #wk
    de_dy=-1*(Y_d[p].reshape((-1,1))-model.evalf(subs=dict_vals))
    wk_vals,wk_grads,wk_local_grads,init_local_grad_j=update_w_grad(phi_k,vk,wk,lj,de_dy,X,mx,wk_vals,dict_vals,eta)
    dict_vals.update(dict(zip(wk,wk_vals)))
    #wj
    wj_vals,wj_grads,wj_local_grads,init_local_grad_i=update_w_grad(phi_j,vj,wj,li,init_local_grad_j,X,mx,wj_vals,dict_vals,eta)
    dict_vals.update(dict(zip(wj,wj_vals)))
    #wi
    wi_vals,wi_grads,wi_local_grads, init_local_grad_0 = update_w_grad(phi_i,vi,wi,Matrix(X[p]),init_local_grad_i,X,mx,wi_vals,dict_vals,eta)
    dict_vals.update(dict(zip(wi,wi_vals)))
    # guardar las funciones
    w_vals.append(list(wi_vals)+list(wj_vals)+list(wk_vals))
    w_grads.append(list(wi_grads)+list(wj_grads)+list(wk_grads))
    w_local_grads.append(list(wi_local_grads)+list(wj_local_grads)+list(wk_local_grads))
  return mi,mj,mk,mx,model,w_local_grads,w_vals,w_grads,dict_vals

def MLP_4capas(X,Y_d,eta):
  # modelo MLP
  # funciones de activacion
  a=1; b=0; v = symbols("v")
  phi_1=a*v+b # lineal
  phi_2=(exp(v)-exp(-v))/(exp(v)+exp(-v)) # tanh
  phi_3=1/(1+exp(-v)) # sigmoid

  # Planteamiento forward
  n=X.shape[1]; mx=Matrix(symarray('X', n)) # input
  mi=2; phi_i=phi_2; li, wi, vi=layer(mx,mi,phi_i,"i")
  mj=2; phi_j=phi_3; lj, wj, vj=layer(li,mj,phi_j,"j")
  mj2=2; phi_j2=phi_3; lj2, wj2, vj2=layer(lj,mj2,phi_j2,"j2")
  mk=Y_d.shape[1]; phi_k=phi_1; lk, wk, vk=layer(lj2,mk,phi_k,"k")
  model, w_vars= lk, list(wi)+list(wj)+list(wj2)+list(wk) # output

  # w inicial aleatoria
  np.random.seed(15)
  wi_vals,wj_vals,wj2_vals,wk_vals=[Matrix(np.random.rand(w.shape[0],w.shape[1])) for w in [wi,wj,wj2,wk]]
  dict_vals=dict(set(zip(wi,wi_vals))|set(zip(wj,wj_vals))|set(zip(wj2,wj2_vals))|set(zip(wk,wk_vals)))

  # calcular gradientes para el punto p y, en cada capa, actualizar w de manera secuencial
  epocas=5
  w_vals=[list(wi_vals)+list(wj_vals)+list(wj2_vals)+list(wk_vals)]; w_local_grads=[]; w_grads=[]
  for iteracion in range(len(X)*epocas):
    p=iteracion%len(X)
    dict_vals.update(dict(set(zip(mx,X[p]))))
    #wk
    de_dy=-1*(Y_d[p].reshape((-1,1))-model.evalf(subs=dict_vals))
    wk_vals,wk_grads,wk_local_grads,init_local_grad_j2=update_w_grad(phi_k,vk,wk,lj2,de_dy,X,mx,wk_vals,dict_vals,eta)
    dict_vals.update(dict(zip(wk,wk_vals)))
    #wj2
    wj2_vals,wj2_grads,wj2_local_grads,init_local_grad_j=update_w_grad(phi_j2,vj2,wj2,lj,init_local_grad_j2,X,mx,wj2_vals,dict_vals,eta)
    dict_vals.update(dict(zip(wj2,wj2_vals)))
    #wj
    wj_vals,wj_grads,wj_local_grads,init_local_grad_i=update_w_grad(phi_j,vj,wj,li,init_local_grad_j,X,mx,wj_vals,dict_vals,eta)
    dict_vals.update(dict(zip(wj,wj_vals)))
    #wi
    wi_vals,wi_grads,wi_local_grads, init_local_grad_0 = update_w_grad(phi_i,vi,wi,Matrix(X[p]),init_local_grad_i,X,mx,wi_vals,dict_vals,eta)
    dict_vals.update(dict(zip(wi,wi_vals)))
    # guardar las funciones
    w_vals.append(list(wi_vals)+list(wj_vals)+list(wj2_vals)+list(wk_vals))
    w_grads.append(list(wi_vals)+list(wj_vals)+list(wj2_vals)+list(wk_vals))
    w_local_grads.append(list(wi_local_grads)+list(wj_local_grads)+list(wj2_local_grads)+list(wk_local_grads))
  return mi,mj,mj2,mk,mx,model,w_local_grads,w_vals,w_grads,dict_vals

# graficar los resultados
def grafica_individual (ax,df,ylabel,title):
  # graficar
  df.plot(ax=ax); ax.set_ylabel(ylabel); ax.set_xlabel("iteracion")
  # dar formato a las figuras
  lines, labels = [sum(x, []) for x in zip(*[ax.get_legend_handles_labels()])]
  ax.legend(lines, labels, loc='upper left', ncol=12, fontsize="small",title=title, alignment='left')

def graficas (w_local_grads,conf,capas_l,numero):
  # dar formato a la informacion guardada
  index=["(p%i, epoca %i)" %(i%len(X), int(i/len(X))) for i in range(len(w_local_grads))]
  name_neuronas=["neurona_%s%i" %(capa, neuronas) for capa in capas_l for neuronas in range(conf[capa])]
  df_local_grads=pd.DataFrame(np.array(w_local_grads),columns=name_neuronas,index=index).astype(float)

  fig, ax = plt.subplots(len(conf),1,figsize=(12,5),sharex=True)
  i=0; df=df_local_grads
  for capa_n, capa in enumerate(conf):
    l_capa=conf[capa]
    grafica_individual(ax[capa_n],df[df.columns[i:i+l_capa]],"$\\delta_l$","capa %s"%capa)
    fig.suptitle("gradientes locales en cada iteracion del descenso por el gradiente")
    i+=l_capa
  fig.tight_layout(); plt.savefig('gradientes_anasofiagutierrez_%i.png'%numero)
  
if __name__=="__main__":
    # Datos
    File_data = np.loadtxt("bacterial_growth.txt", dtype=float, delimiter=',')
    X = File_data[:100,[0,1]]
    Y_d = File_data[:100,[-1]]
    File_data = File_data/np.max(File_data,axis=0) # normalizar a [-1,1]
    X_norm = File_data[:100,[0,1]]
    Y_d_norm = File_data[:100,[-1]]

    # aprendizaje: 3 capas, eta=1
    mi,mj,mk,mx,model,w_local_grads1,w_vals1,w_grads1,dict_vals1=MLP_3capas(X,Y_d,eta=1)
    conf={'i':mi,'j':mj,'k':mk}
    capas_l=['i','j','k']
    graficas (w_local_grads1,conf,capas_l,1)
    mi,mj,mk,mx,model,w_local_grads2,w_vals2,w_grads2,dict_vals2=MLP_3capas(X_norm,Y_d_norm,eta=1)
    graficas (w_local_grads2,conf,capas_l,2)

    # aprendizaje: 3 capas, eta=0.5
    mi,mj,mk,mx,model,w_local_grads3,w_vals3,w_grads3,dict_vals3=MLP_3capas(X,Y_d,eta=0.5)
    graficas (w_local_grads3,conf,capas_l,3)
    mi,mj,mk,mx,model,w_local_grads4,w_vals4,w_grads4,dict_vals4=MLP_3capas(X_norm,Y_d_norm,eta=0.5)
    graficas (w_local_grads4,conf,capas_l,4)

    # aprendizaje: 4 capas
    mi,mj,mj2,mk,mx,model,w_local_grads5,w_vals5,w_grads5,dict_vals5=MLP_4capas(X,Y_d,eta=0.5)
    conf2={'i':mi,'j':mj,'j2':mj2,'k':mk}
    capas_l2=['i','j','j2','k']
    graficas(w_local_grads5,conf2,capas_l2,5)
    mi,mj,mj2,mk,mx,model,w_local_grads6,w_vals6,w_grads6,dict_vals6=MLP_4capas(X_norm,Y_d_norm,eta=0.5)
    graficas(w_local_grads6,conf2,capas_l2,6)
    
  