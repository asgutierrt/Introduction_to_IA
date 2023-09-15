# instalar e importar las librerias necesarias
try:
    import os
    from mlp_functions import *
    from format_support import *
except:
    import os
    cmds=['pip install --upgrade pip', 'pip install numpy sympy pandas matplotlib tqdm sys time']
    for cmd in cmds: os.system(cmd)
 
if __name__=="__main__":
############## 
    # setup -> paths, data, initialize variables
    ## paths to save results
    curr_path=os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(curr_path, r'figures')
    csv_path = os.path.join(curr_path, r'resultados')
    if not os.path.exists(fig_path): os.makedirs(fig_path)
    if not os.path.exists(csv_path): os.makedirs(csv_path)

    ## load data
    data_path="final.txt"
    X,Y_d,X_test,Y_d_test,X_val,Y_d_val=load_data(data_path)

    ## activation functions
    a=1; b=0; v = symbols("v")
    phi_1=a*v+b # lineal
    phi_tanh=(exp(v)-exp(-v))/(exp(v)+exp(-v)) # tanh
    phi_sigmoid=1/(1+exp(-v)) # sigmoid

    ## initialize dfs to save results' data
    model_label_info=['layers', 'neurons', 'eta']
    error=pd.DataFrame(columns=model_label_info).set_index(model_label_info)
    energia_error=pd.DataFrame(); avrg_delta=pd.DataFrame(); 
    deltas=pd.DataFrame(); predictions=pd.DataFrame()
    w_vals=pd.DataFrame(columns=model_label_info).set_index(model_label_info).astype(object)
    
    ''' to load previous files
    import ast
    error=pd.read_csv("errores_train_test.csv",index_col=[0,1,2])
    energia_error=pd.read_csv("energia_errores.csv",header=[0,1,2,3],index_col=0)
    avrg_delta=pd.read_csv("avrg_deltas.csv",header=[0,1,2,3],index_col=0)
    predictions=pd.read_csv("predictions.csv",index_col=0)
    predictions.columns=[ast.literal_eval(x) for x in predictions.columns]
    predictions=predictions.applymap(lambda x: [float(val) for val in x.strip('][').split(',')])
    w_vals=pd.read_csv("w_vals.csv",index_col=[0,1,2]).applymap(lambda x: sympify(x) if type(x)!=float else np.nan)
    w_vals.columns=[int(x) for x in w_vals.columns]
    '''
    

##############
    # experimentation
    epocas=4; tol=10**-2
    n=len(X[0]); m=len(Y_d[0])
    dfs=[predictions,w_vals,energia_error,error,deltas,avrg_delta] # pack dfs
    for L in [1,2,3]:
        for eta in [0.9,0.5,0.2]:
            exp_name="L=%i eta=%.2f"%(L,eta) # experiment identifier
            print('running ',exp_name)
            for mj in [1,2,3,4,5]:
                model_label=('L=%i'%L, '$l_i$=%i'%mj,'$eta=%.2f$'%eta)
                start=time.time()
                # initialize model
                model=MLP([(n,phi_sigmoid)]+[(mj,phi_sigmoid)]*L+[(m,phi_sigmoid)])# init model on random sol
                # gradient descent
                local_grads,avrg_error,grad_dead_at=grad_descent(X,Y_d,model,epocas=epocas,eta=eta,tol=tol)
                # save training information
                dfs=process_exp(model_label,model,local_grads,avrg_error,grad_dead_at,
                                n,L,start,X_test,Y_d_test,dfs,csv_path)
            # plot experiment's figures
            args_exp={'key':("L=%i"%L,'$eta=%.2f$'%eta), 'level':[1,3],'axis':1}
            avrg_delta=dfs[-1] # unpack avrg_deltas information
            graficar_df(avrg_delta.xs(**args_exp),'v',lambda i: "avrg $\\delta_%s$"%i,
                        'iteration (0-N*epochs)',"average local gradient progression - %s"%exp_name,
                        rot=0,figsize=(12,6),name='avrg_delta_%s'%exp_name.replace(" ", "_"),path=fig_path)

##############
    # unpack experiment's data
    predictions,w_vals,energia_error,error,deltas,avrg_delta=dfs

    # compare results for best, average and worst models
    ## selection criteria
    criteria=['test error']
    figs=['best mlp','worst mlp','avrg mlp']
    error.sort_values(by=criteria, inplace=True)
    model_names=[error[error[criteria]==error[criteria].min()].dropna(how='all').index[0],
                error[error[criteria]==error[criteria].max()].dropna(how='all').index[0],
                error[error[criteria] >= error[criteria].mean()].dropna(how='all').iloc[0].name]
    
    ## plot outputs
    graficar_Y(Y_d_test,predictions,figs,model_names,
            "test dataset: real output vs. MLP output",fig_path+'/fig_outputs.png')
    ## plot error energy
    filter=[x[1:] in model_names for x in energia_error.columns]
    graficar_df(energia_error.loc[:,filter],'v',lambda i: "$\\varepsilon_{avg}$",
                'iteration (0-N*epochs)',"Error energy progression",rot=0,
                name='error_energy',path=fig_path)
    ## plot avrg local gradients
    filter=[x[1:] in model_names for x in avrg_delta.columns]
    graficar_df(avrg_delta.loc[:,filter],'v',lambda i: "avrg $\\delta_%s$"%i,
                'iteration (0-N*epochs)',"average local gradient progression",
                rot=0,figsize=(12,6),name='avrg_delta',path=fig_path)
    '''
    ## validation errors
    for model_label in model_names:
        L=int(model_label[0][-1]); mj=int(model_label[1][-1])
        eta=float(model_label[2].rsplit("=")[1].split('$')[0])
        # model=MLP([(n,phi_sigmoid)]+[(mj,phi_sigmoid)]*L+[(m,phi_sigmoid)])
        # error en validacion
        model.sympy_frwd_eval(w_vals.loc[model_label].dropna().to_dict(),n)
        Y_pred=[list(model.conf[L+1].Y.xreplace(dict(zip(model.conf[L+1].Y.free_symbols,xp)))) for xp in X_val]
        val_errors=np.mean([sum(e**2)/2 for e in (Y_d_val-Y_pred)])
        error.loc[model_label,'val error']=val_errors
    error.dropna(subset=['val error']).to_csv(csv_path+"/errores_val.csv")
    '''
    