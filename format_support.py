try:
    from matplotlib import pyplot as plt
    import pandas as pd
    import time
    import numpy as np
except:
    import os
    cmds=['pip install --upgrade pip', 'pip install pandas matplotlib time numpy']
    for cmd in cmds: os.system(cmd)

def load_data(data_path):
    File_data = np.loadtxt(data_path, dtype=float, delimiter=',')
    File_data = File_data/np.max(File_data,axis=0) # normalizar a [0,1]

    # muestrear
    n_data=len(File_data)
    ix=np.random.choice(n_data, n_data, replace=False)
    n_train=int(n_data*0.6); n_test=int(n_data*0.2); n_val=int(n_data*0.2)

    # sort data
    data_train=File_data[ix[:n_train]]; 
    data_train_sorted=File_data[np.sort(ix[:n_train])]
    data_test=File_data[np.sort(ix[n_train:n_train+n_test])]
    data_val=File_data[np.sort(ix[-n_val:])]

    # split data
    X = data_train[:,[0,1]]; Y_d = data_train[:,[2]]
    X_sorted = data_train_sorted[:,[0,1]]; Y_d_sorted = data_train_sorted[:,[2]]
    X_test=data_test[:,[0,1]]; Y_d_test = data_test[:,[2]]
    X_val=data_val[:,[0,1]]; Y_d_val = data_val[:,[2]]

    return X,Y_d,X_test,Y_d_test,X_val,Y_d_val

def guardar_informacion(model_label,csv_path,avrg_error,local_grads,test_errors,dfs,start,grad_dead_at):
    energia_error,error,deltas,avrg_delta=dfs
    # guardar informacion
    end=time.time()
    # energia error
    df_error=pd.DataFrame(avrg_error,columns=pd.MultiIndex.from_tuples([('error',)+model_label]),dtype=float)
    energia_error=pd.concat([energia_error,df_error], axis=1); energia_error.to_csv(csv_path+"/energia_errores.csv")

    # error en entrenamiento
    error.loc[model_label,'train error']=df_error.mean().values[0]

    # error en test
    error.loc[model_label,'test error']=test_errors
    error.loc[model_label,'time [s]']=end-start

    # local gradients
    cols_label=lambda layer, neurons: pd.MultiIndex.from_tuples([("$\\delta_%s$"%layer,)+model_label+('neuron %i'%i,) for i in neurons])
    info_delta=lambda grad_info,layer: pd.DataFrame(grad_info,dtype=float,columns=cols_label(layer,range(len(grad_info[0]))))
    df_local_grads=pd.concat([info_delta(local_grads[layer],layer) for layer in pd.DataFrame(local_grads)],axis=1)
    deltas=pd.concat([deltas,df_local_grads], axis=1); deltas.to_csv(csv_path+"/deltas.csv")

    # mean local gradients
    df_mean=df_local_grads.groupby(level=[0,1,2,3],axis=1).mean()
    avrg_delta=pd.concat([avrg_delta,df_mean], axis=1); avrg_delta.to_csv(csv_path+"/avrg_deltas.csv")

    # estabilizacion
    df_stable_mean=df_local_grads.iloc[grad_dead_at:].groupby(level=[0],axis=1).mean()
    error.loc[model_label,df_stable_mean.columns]=df_stable_mean.mean(axis=0)[df_stable_mean.columns]
    error.loc[model_label,'gradient death i']=grad_dead_at
    error.to_csv(csv_path+"/errores_train_test.csv")
    
    return energia_error, error, deltas, avrg_delta

def process_exp(model_label,model,local_grads,avrg_error,grad_dead_at,n,L,start,X_test,Y_d_test,dfs,csv_path):
    # unpack dfs
    predictions,w_vals,energia_error,error,deltas,avrg_delta=dfs
    # error en test
    model.sympy_frwd_eval(model.w_dict,n)
    Y_pred=[list(model.conf[L+1].Y.xreplace(dict(zip(model.conf[L+1].Y.free_symbols,xp)))) for xp in X_test]
    test_errors=np.mean([sum(e**2)/2 for e in (Y_d_test-Y_pred)])
    predictions[model_label]=Y_pred
    predictions.to_csv(csv_path+"/predictions.csv")
    # guardar informacion del entrenamiento
    w_vals.loc[model_label,model.w_dict.keys()]=model.w_dict.values()
    w_vals.to_csv(csv_path+"/w_vals.csv")
    energia_error,error,deltas,avrg_delta=guardar_informacion(model_label,csv_path,
                                                                avrg_error,local_grads,test_errors,
                                                                [energia_error,error,deltas,avrg_delta],
                                                                start,grad_dead_at)
    return predictions,w_vals,energia_error,error,deltas,avrg_delta

def graficar_df(df,layout,y_label,x_label,title,rot=0,figsize=(7,4),name='',path=''):
    dim_names=set(df.columns.get_level_values(0))
    fig_x,fig_y= (1,len(dim_names)) if layout=="h" else (len(dim_names),1)
    fig, ax = plt.subplots(fig_x,fig_y,figsize=figsize,sharex=True,squeeze=False)
    labels=None
    for i, (dimension, df_) in enumerate(df.groupby(level=0,axis=1)):
        ix,iy= (0,i) if layout=="h" else (i,0)
        labels_i=df_[dimension].columns.tolist()
        df_.plot(ax=ax[ix,iy],style=["-"]*(len(labels_i)-1)+['-'],rot=rot,legend=False)
        ax[ix,iy].set_ylabel(y_label(i))
        ax[ix,iy].set_xlabel(x_label)
        if labels is None: labels=labels_i
    fig.suptitle(title,y=1)
    fig.legend(labels,bbox_to_anchor=[0.5, 0.97],loc='upper center',ncol=12,fontsize="small",alignment='left')
    fig.tight_layout(); plt.savefig(path+'/fig_%s.png'%name)

def graficar_Y(Y_d_test,predictions,figs,model_names,title,fig_path):
    fig = plt.figure(figsize=(8,5))
    plt.plot(Y_d_test)
    for name in model_names:
        Y_pred=predictions[name]; output_k=0
        plt.plot(Y_pred.str[output_k])
    fig.legend(['Y real']+[model+" "+str(name) for model,name in zip(figs,model_names)],
            bbox_to_anchor=[0.98, 0.899],loc='upper right')
    fig.suptitle(title);fig.tight_layout(); plt.savefig(fig_path)