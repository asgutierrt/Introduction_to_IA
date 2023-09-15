# instalar e importar las librerias necesarias
try:
    import numpy as np
    from sympy import *
    from tqdm import tqdm
except:
    import os
    cmds=['pip install --upgrade pip', 'pip install numpy sympy tqdm']
    for cmd in cmds: os.system(cmd)

class MLP:
  __slots__ = ['conf','w_dict']
  def __init__(self,layers_setup):
    self.conf={}; self.w_dict={}; entrada=None
    for layer,(neuronas,phi) in enumerate(layers_setup):
      if entrada is None: entrada=Matrix(symarray('X', neuronas))
      self.conf[layer]=layer_obj(entrada,neuronas,phi)
      self.w_dict[layer]=self.conf[layer].W
      entrada=self.conf[layer].Y

  def frwd_eval(self,w_dict,x):
    entrada=x
    for layer in range(len(self.conf)):
      self.conf[layer].update_w(w_dict[layer],entrada)
      entrada=self.conf[layer].Y

  def sympy_frwd_eval(self,w_dict,n_x):
    entrada=Matrix(symarray('X', n_x))
    for layer in range(len(self.conf)):
      self.conf[layer].update_w(w_dict[layer],entrada)
      entrada=self.conf[layer].Y

class layer_obj:
  __slots__ = ['m','phi','x','W','V','Y','dY_dV']
  def __init__(self,x,m,phi):
    self.m=m; self.phi=phi # estaticos
    self.x=x; self.W=Matrix(np.random.rand(m,int(x.shape[0]))*np.sqrt(2/m)) # inicializar variables w para la capa
    self.frwd_eval(m,phi)

  def frwd_eval(self,m,phi):
    self.V=self.W*self.x # campo local inducido en cada perceptron
    # funcion de activacion evaluada en el campo de los m perceptrones
    self.Y=Matrix(m,1, lambda i,j: phi.subs({"v":self.V[i,j]}))
    # derivada de la funcion de activacion evaluada en el campo
    self.dY_dV=Matrix(m,1, lambda i,j: diff(phi,"v").subs({"v":self.V[i,j]}))

  def update_w(self,w,x):
    self.W=w; self.x=x
    self.frwd_eval(self.m,self.phi)

def grad_descent(X,Y_d,model,epocas=5,eta=1,tol=0):
  layers_labels=sorted(model.conf.keys(),reverse=True) # orden para backpropagation
  # almacenar informacion de cada iteracion
  local_grads={layer:[] for layer in layers_labels}; avrg_error=[]
  pbar = tqdm(total=len(X)*epocas*len(layers_labels)) # create progress bar
  iteracion=0; grad_dead_at=None
  while iteracion < len(X)*epocas:
    p=iteracion%len(X); iteracion+=1
    model.frwd_eval(model.w_dict,X[p]) # forward evaluation
    # backward evaluation
    init_local_grad=None
    for layer in layers_labels:
      # calculate gradient
      if init_local_grad is None:
        e=Y_d[p].reshape((-1,1))-model.conf[layer].Y
        avrg_error.append((e.T*e)[0]/2)
        init_local_grad=-1*e
      local_grad=init_local_grad.multiply_elementwise(model.conf[layer].dY_dV)
      dJ_dw=local_grad*model.conf[layer].x.T
      init_local_grad=model.conf[layer].W.T*local_grad
      # update w vals in layer
      model.w_dict[layer]+=eta*dJ_dw
      # save local gradients
      local_grads[layer].append(list(local_grad))
      pbar.update(1) # update progress bar

    # guardar en que iteracion muere el gradiente
    window_size=int(len(X)*0.25)
    if grad_dead_at is None and iteracion>window_size and all([(np.absolute(np.diff(np.mean(local_grads[layer][-window_size:],axis=1),n=1))<tol).all() for layer in layers_labels]):
      grad_dead_at=iteracion
    # detenerse solo despues de la primera epoca
    if grad_dead_at and iteracion>grad_dead_at+window_size:
      pbar.close()
      print("gradients died first at iteration %i"%grad_dead_at)
      break
  return local_grads, avrg_error, grad_dead_at

