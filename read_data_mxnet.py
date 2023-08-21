import pandas as pd
import numpy as np
from glob import glob

def create_data_frame(file_name,name):
    df=pd.read_csv('Extendido/'+file_name)
    df['FECHA']=df['FECHA'].astype(str)
    df['HORA']=df['HORA'].astype(str)
    df['YEAR']=df['FECHA'].apply(lambda x: '20'+x[:2]+'-'+x[2:4]+'-'+x[-2:])
    df['HOUR']=df['HORA'].apply(lambda x : '00:00' if len(x)==1 else  '0'+x[0]+':'+x[-2:] if len(x)==3 else x[0:2]+':'+x[-2:])
    df['DATETIME']=pd.to_datetime(df['YEAR']+' '+df['HOUR'])
    df.drop(columns=['FECHA','HORA','YEAR','HOUR'],inplace=True)
    df.set_index('DATETIME',inplace=True)
    df['item_id']=name
    return df

def create_timeseries_tensor(df,context_length,prediction_legth):
    X_new=list()
    y_new=list()
    for i in range(0,df.shape[0]-(context_length+prediction_legth)):
        batch=df[i:i+context_length+prediction_legth]
        if ~batch.isnull().values.any():
            X_new.append(batch[:context_length].to_numpy())
            y_new.append(batch[context_length:].to_numpy().ravel())
    X=np.stack([np.array(x) for x in X_new],axis=0)
    y=np.stack([np.array(y) for y in y_new],axis=0)
    return X_new,y_new

###train##            

def create_dataset_pull_train(context_length,prediction_length):
    files=['utal_pm25_ia.csv','ucm_pm25_ia.csv','lf_pm10_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_train,y_train)

def create_dataset_ucm_train(context_length,prediction_length):
    files=['ucm_pm25_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_train,y_train)

def create_dataset_utal_train(context_length,prediction_length):
    files=['utal_pm25_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_train,y_train)

def create_dataset_lf_train(context_length,prediction_length):
    files=['lf_pm10_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_train,y_train)

####test####
def create_dataset_pull_test(context_length,prediction_length):
    files=['utal_pm25_ia.csv','ucm_pm25_ia.csv','lf_pm10_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_test,y_test)

def create_dataset_ucm_test(context_length,prediction_length):
    files=['ucm_pm25_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_test,y_test)

def create_dataset_utal_test(context_length,prediction_length):
    files=['utal_pm25_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_test,y_test)

def create_dataset_lf_test(context_length,prediction_length):
    files=['lf_pm10_ia.csv']
    df_list=[create_data_frame(f,f.split('_')[0]) for f in files]
    df=pd.concat(df_list)
    df.drop(df.columns.difference(['PM2_5','item_id','FECHA']), 1, inplace=True)

    dfs_train = {}
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        #gdf.set_index('fecha',drop=True,inplace=True)
        gdf=gdf.resample('1H').mean()
        idx_train=round(len(gdf)*.8)
        dfs_train[item_id] = gdf.iloc[:idx_train]
        dfs_test[item_id] = gdf.iloc[idx_train:]



    X_train=list()
    y_train=list()
    for site in dfs_train:
        X,y=create_timeseries_tensor(dfs_train[site],context_length,prediction_length)
        X_train.append(X)
        y_train.append(y)
    X_train=np.concatenate(X_train,axis=0)
    y_train=np.concatenate(y_train,axis=0)
    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_test,y_test)
#########

def create_grid_test():
    files=[pd.read_json(f_name,lines=True) for f_name in glob('data/*.json')]
    df=pd.concat(files)
    df.reset_index(inplace=True,drop=True)
    df['fecha']=pd.to_datetime(df['dt_u'], unit='s')
    df.to_pickle("./datos_calidad_aire.pkl") 

def create_dataset_test(context_length,prediction_length):
    df=pd.read_pickle('./datos_calidad_aire.pkl')
    df['fecha']=pd.to_datetime(df['dt_u'], unit='s')
    df['item_id'] = list(zip(df.lat, df.lon))
    df.drop(df.columns.difference(['pm2_5','item_id','fecha']), 1, inplace=True)
    dfs_test = {}
    for item_id, gdf in df.groupby("item_id"):
        if not ( pd.isna(item_id[0]) or pd.isna(item_id[1]) ):
            gdf.set_index('fecha',drop=True,inplace=True)
            gdf=gdf.resample('1H').mean()
            dfs_test[item_id] = gdf.resample('1H').mean()    
    X_test=list()
    y_test=list()
    for site in dfs_test:
        X,y=create_timeseries_tensor(dfs_test[site],context_length,prediction_length)
        X_test.append(X)
        y_test.append(y)
    X_test=np.concatenate(X_test,axis=0)
    y_test=np.concatenate(y_test,axis=0)
    return (X_test,y_test)

