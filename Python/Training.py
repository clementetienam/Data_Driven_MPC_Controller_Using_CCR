# -*- coding: utf-8 -*-
"""
ABC-RP: Data Driven MPC
Created on Saturday Nov  17 09:39:40 2020
@author: Dr Clement Etienam
@Line Manager: Dr Joshua Sykes
@Research Associate: Siying Shen
------------------------------------------------------------------------------
Box Configuration
Inputs:
'Ambient temperature (C)'
'Ground temperature (C)'
'Global irradiance (W/m2)'
'Direct irradiance (W/m2)'
'Diffuse irradiance (W/m2)'
'Heat pump heat supply (kW)'
'Heat pump electrical load (kW)'

Outputs:
'Internal temperature (C) '


Data Driven MPC Approach. Online approach
steps:
    1) Predict room temperature at time t given current weather states
    2) Optimise for control at time t to reference room temperature
    3) Predict room temperature at time t+1 using temperature at time t
    4) Predict weather states for t+1 using temperature at t+1 (gotten from 3)
    4) Set for next evolution, temperature at t= temperature at t+1
      (prior(t)= posterior(t+1))
    
mathematically;    
y=room temperature
X=weather states
u=control for GSHP pump

input: u(t-1)= initial guess,X(t-1)(= Known), r for all t(= known), f1,g (= Learned),
y(t-1)(=Infered from y(t-1)=f1(X(t-1))+e )
g=Xgboost machine
f2=Augmented states (with control inputs) to room temperature

set:
y(1)=y(t-1) 
X(1)=X(t-1) 
u(1)=u(t-1) 
Do t= 1: Horizon:
    y(t+1)=g(y(t))+n # Predict the future output given present output
    y(t)=f1(X(t))+e # Predict current output with current states
    ybig(t,:)=y(t)
    u(opt)=argmin||r(t)-f2(X(t);u(t),X(t))||+z # Optimise the control at time t
    ubig(t,:)=u(opt)
    Xbig(t,:)=X
    
    X(opt)=argmin||y(t+1)-f1(X(t)||+z # Optimise the state at time t+1
    set X(t)= X(t+1)=X(opt)
    set u(t)= u(t+1)=u(opt)
    
    
End Do

Time series machine: XGboost
Forward mapping machine: RandomForest

"""
from __future__ import print_function
print(__doc__)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.stats import rankdata, norm
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import datetime 
import multiprocessing
import os
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
import seaborn as sns
import numpy
import shutil
from copy import copy
from joblib import Parallel, delayed
import xgboost as xgb
from xgboost import plot_importance
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import os
from kneed import KneeLocator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import load_model
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
## This section is to prevent Windows from sleeping when executing the Python script
class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        import ctypes
        #Preventing Windows from going to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS | \
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        import ctypes
        #Allowing Windows to go to sleep
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)
osSleep = None
# in Windows, prevent the OS from sleeping while we run
if os.name == 'nt':
    osSleep = WindowsInhibitor()
    osSleep.inhibit()
##------------------------------------------------------------------------------------

oldfolder = os.getcwd()
cores = multiprocessing.cpu_count()
print(' ')
print(' This computer has %d cores, which will all be utilised in parallel '%cores)
#print(' The number of cores to be utilised can be changed in runeclipse.py and writefiles.py ')
print(' ')

start = datetime.datetime.now()
print(str(start))

print('-------------------LOAD FUNCTIONS-------------------------------------')
def interpolatebetween(xtrain,cdftrain,xnew):
    numrows1=len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2=np.zeros((numrows1,numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:,i]), cdftrain[:,i],kind='linear')
        cdftest = f(xnew[:,i])
        norm_cdftest2[:,i]=np.ravel(cdftest)
    return norm_cdftest2


def gaussianizeit(input1):
    numrows1=len(input1)
    numcols = len(input1[0])
    newbig=np.zeros((numrows1,numcols))
    for i in range(numcols):
        input11=input1[:,i]
        newX = norm.ppf(rankdata(input11)/(len(input11) + 1))
        newbig[:,i]=newX.T
    return newbig
	
def getoptimumk(X,i,training_master,oldfolder):
#    X=matrix
    distortions = []
    Kss = range(1,10)
    
    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    #ncusters2=np.where(distortions == distortions.min())
    
    myarray = np.array(distortions)
    
    knn = KneeLocator(Kss,myarray,curve='convex',direction='decreasing',interp_method='interp1d')
    kuse=knn.knee
    
    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, 'bx-')
    plt.xlabel('cluster size')
    plt.ylabel('Distortion')
    plt.title('Elbow Method showing the optimal n_clusters for machine %d'%(i))
    os.chdir(training_master)
    plt.savefig("machine_%d.jpg"%(i+1))
    os.chdir(oldfolder)
    plt.show()
    return kuse	
	
	
    
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
    return a, b
def Performance_plot(CCR,Trued,stringg,training_master,oldfolder):
    print(' Compute L2 and R2 for the machine')
    clementanswer2=np.reshape(CCR[:,0],(-1,1))
    outputtest2=np.reshape(Trued[:,0],(-1,1))
    numrowstest=len(outputtest2)    
    print('For output 2-THERMAL ZONE: BOX:Zone Air Temperature [C](10 Minutes) ')
    outputtest2 = np.reshape(outputtest2, (-1, 1))
    Lerrorsparse=(LA.norm(outputtest2-clementanswer2)/LA.norm(outputtest2))**0.5
    L_22=1-(Lerrorsparse**2)
    #Coefficient of determination
    outputreq=np.zeros((numrowstest,1))
    for i in range(numrowstest):
    	outputreq[i,:]=outputtest2[i,:]-np.mean(outputtest2)
    CoDspa=1-(LA.norm(outputtest2-clementanswer2)/LA.norm(outputreq))
    CoD2=1 - (1-CoDspa)**2 ;
    print('')        	
    CoDoverall=(CoD2)/1    
    R2overall=(L_22)/1      
    CoDview=np.zeros((1,1))
    R2view=np.zeros((1,1))
    CoDview[:,0]=CoD2
    R2view[:,0]=L_22
    plt.figure(figsize =(20,20))
    palette = copy(plt.get_cmap('inferno_r'))
    palette.set_under('white')  # 1.0 represents not transparent
    palette.set_over('black')  # 1.0 represents not transparent
    vmin=min(np.ravel(outputtest2))
    vmax=max(np.ravel(outputtest2))
    sc=plt.scatter(np.ravel(clementanswer2),np.ravel(outputtest2),c=np.ravel(outputtest2), vmin=vmin, vmax=vmax, s=35, cmap=palette)
    plt.colorbar(sc)
    plt.title('Temperature [C](15 Minutes)', fontsize = 20)
    plt.ylabel('Machine',fontsize = 20)
    plt.xlabel('True data',fontsize = 20)
    a,b=best_fit(np.ravel(clementanswer2), np.ravel(outputtest2),)
    yfit = [a + b * xi for xi in np.ravel(clementanswer2)]
    plt.plot(np.ravel(clementanswer2), yfit,color='r')
    plt.annotate('R2= %.3f' % CoD2, (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=20)    
    os.chdir(training_master)        
    plt.savefig("%s.jpg"%stringg)
    os.chdir(oldfolder)
    #plt.show()
	
    return CoDoverall,R2overall,CoDview,R2view  


def parad2(numruth,X_train,numgeh,y_traind,training_master,oldfolder):
    
    np.random.seed(7)
   
    a0=X_train
    a0=np.reshape(a0,(-1,numruth),'F')

    b0=y_traind
    b0=np.reshape(b0,(-1,numgeh),'F')
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    degree=9
    model=make_pipeline(PolynomialFeatures(degree),scaler,LinearRegression())
  
    model.fit(a0, b0)
    os.chdir(training_master)
    pickle.dump(model, open("Machine.asv", 'wb'))    
    os.chdir(oldfolder)
    #return modelDNN


def Read_data_csv(Name,training_master,oldfolder,stringg): # Name is string of xsls file
    df=pd.read_csv(Name)
    print(df.head())
    for col in df.columns: 
        print(col)
    print('')
    print('')
    data1=df
    for col1 in data1.columns: 
        print(col1)        
    data1=data1.dropna()
    corr=data1.corr()
    plt.figure(figsize=(14,8))
    plt.title('correlation of data',fontsize=25)
    ax=sns.heatmap(corr, vmin=corr.values.min(), vmax=1, square=True, cmap="YlGnBu", linewidths=0.1, annot=True, annot_kws={"fontsize":8}) 
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    os.chdir(training_master)
    plt.savefig("%s.jpg"%stringg)
    os.chdir(oldfolder)
    plt.show
    data1=data1.values
    temp=data1
    outpuut=temp[:,[7]]
    inpuut=temp[:,[0,1,2,3,4,5,6]]
    return inpuut,outpuut ,df

"""
if there is a sensor, then use this as time t and use the time series model to predict the next!
"""
def create_features(df,datetime_series, label):
    """
    Creates time series features from datetime index
    """
    df['Date/Time']=datetime_series
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
    df['date'] = df.index
    df['hour'] = df['Date/Time'].dt.hour
    df['dayofweek'] = df['Date/Time'].dt.dayofweek
    df['quarter'] = df['Date/Time'].dt.quarter
    df['month'] = df['Date/Time'].dt.month
    df['year'] = df['Date/Time'].dt.year
    df['dayofyear'] = df['Date/Time'].dt.dayofyear
    df['dayofmonth'] = df['Date/Time'].dt.day
    df['weekofyear'] = df['Date/Time'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X



def Time_series_learning(df,spitt,datetime_series,ii,\
                         training_master,oldfolder,train_size):

    df_train = df[:train_size]
    
    df_test = df[train_size:]
    
    
    X_train, y_train = create_features(df_train,datetime_series, label=\
                                       spitt)
    
    
    X_test, y_test = create_features(df_test, datetime_series,label=\
                                       spitt)
    print('-----------------------Learn the Model-----------------------------')
    reg = xgb.XGBRegressor(n_estimators=4000,max_depth = 2000,colsample_bytree = 0.95,learning_rate = 0.1)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), \
                      (X_test, y_test)],
            early_stopping_rounds=1500,
           verbose=False)
    filename="series" + str(ii) +".asv"
    os.chdir(training_master)

    pickle.dump(reg, open(filename, 'wb'))        
        
     
    pltk = plot_importance(reg, height=0.9)
    pltk.figure.savefig("%d.jpg"%ii)
    os.chdir(oldfolder)
    plt.show()
    
    aa= reg.predict(X_test)
    bb=y_test.values
    
    
    future1 = [None] * len(aa) 
    future1 = pd.DataFrame(future1)
    future1.columns = ['Date/Time']
    jj2=datetime_series[train_size:]
    jj2=jj2.to_frame()
    future1['Truee']=bb
    future1['Prediction_test']=aa
    
    
    df_test['Prediction'] = reg.predict(X_test)
    pjme_all = pd.concat([df_test, df_train], sort=False)
    pltk2 = pjme_all[[spitt,'Prediction']].plot(figsize=(16, 5))
    filenamez=str(ii)+'b' +".jpg"
    os.chdir(training_master)
    pltk2.figure.savefig("%s.jpg"%filenamez)
    os.chdir(oldfolder)
	
def run_model(model,inn,ouut,i,training_master,oldfolder):
    # build the model on training data
    model.fit(inn, ouut )
    filename='Classifier_%d.asv'%i
    os.chdir(training_master)
    pickle.dump(model, open(filename, 'wb'))
    os.chdir(oldfolder)
    return model

def run_model2(numruth,inn,ouut,training_master,oldfolder,filee):
    np.random.seed(7)
    os.chdir(training_master)
    modelDNN = Sequential()
    modelDNN.add(Dense(200, activation = 'relu', input_dim = numruth))
    modelDNN.add(Dense(units = 820, activation = 'relu'))
    modelDNN.add(Dense(units = 220, activation = 'relu')) 
    modelDNN.add(Dense(units = 21, activation = 'relu'))
    modelDNN.add(Dense(units = 1))
    modelDNN.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=100)
    mc = ModelCheckpoint(filee, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    a0=inn
    a0=np.reshape(a0,(-1,numruth),'F')
    b0=ouut
    b0=np.reshape(b0,(-1,1),'F')
    gff=len(a0)//50
    if gff<1:
        gff=1
    modelDNN.fit(a0, b0,validation_split=0.01, batch_size =100, \
                  epochs = 3000,callbacks=[es,mc])
    
    os.chdir(oldfolder)
    return modelDNN
  
def CCR_Machine(inpuutj,outputtj,ii,training_master,oldfolder,raw_signal):
    print(' Learn the classifer from the predicted labels from Kmeans')
   # model = RandomForestClassifier(n_estimators=50,n_jobs=-1)
    model=xgb.XGBClassifier(n_estimators=1000)
    import numpy as np
    import pickle
    print('cluster with X and y')
    X=inpuutj
    y=outputtj
    numruth = len(X[0])    
    yruth=y
    y_traind=y
    y_traind=numruth*10*y_traind
    matrix=np.concatenate((X,y_traind), axis=1)
    #matrix=raw_signal
    k=getoptimumk(matrix,ii,training_master,oldfolder)
    #k=7
    nclusters=k
    print ('Optimal k is: ', nclusters)
    print('Do the K-means clustering with specified clusters of [X,y] and get the labels')
    kmeans =MiniBatchKMeans(n_clusters=nclusters,max_iter=1000).fit(matrix)
    dd=kmeans.labels_
    dd=dd.T
    dd=np.reshape(dd,(-1,1))
    #-------------------#---------------------------------#
    print('Use the labels to train a classifier')
    inputtrainclass=X
    #inputtrainclass=X
    outputtrainclass=np.reshape(dd,(-1,1))
    modelc=run_model(model,inputtrainclass,outputtrainclass,ii,training_master,oldfolder)
    #modelc=run_model2(numruth,inputtrainclass,outputtrainclass,training_master,oldfolder,'Classifier_%d.h5'%ii)
    dduse = modelc.predict(inputtrainclass)
    #dduse = modelc.predict(inputtrainclass)
    #dduse=np.argmax(dduse, axis=-1) 
        
    print('Split for classifier problem')
    X_train=X
    y_train=dd
    #-------------------Regression----------------#
    print('Learn regression of the clusters with different labels from k-means ' )
    print('set the output matrix')
#    clementanswer=np.zeros((numrowstest,1))
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    degree=9
    
    for i in range(nclusters):
        label0=(np.asarray(np.where(y_train == i))).T
        #model0 = RandomForestRegressor(n_estimators=50,n_jobs=-1)
        model0=make_pipeline(PolynomialFeatures(degree),scaler,LinearRegression())
        #model0=xgb.XGBRegressor(n_estimators=500)
		
        a0=X_train[label0,:]
        a0=np.reshape(a0,(-1,numruth),'F')
#        zz=np.concatenate((ii, i), axis=0)
        b0=yruth[label0,:]
        b0=np.reshape(b0,(-1,1),'F')
        if a0.shape[0]!=0 and b0.shape[0]!=0:
            model0.fit(a0, b0)
        filename="Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) +".asv"
        os.chdir(training_master)

        pickle.dump(model0, open(filename, 'wb'))
        os.chdir(oldfolder)

    return nclusters

 
def PREDICTION_CCR__MACHINE(ii,nclusters,inputtest,numcols,scalerz,training_master,oldfolder):
    import numpy as np
    import pickle
    filename1='Classifier_%d.asv'%ii
    os.chdir(training_master)
    loaded_model = pickle.load(open(filename1, 'rb'))
    os.chdir(oldfolder)
    labelDA = loaded_model.predict(inputtest)
    numrowstest=len(inputtest)

    clementanswer=np.zeros((numrowstest,1))
    
    #print('Start the regression')
    for i in range(nclusters):
#        filename2='Regressor_Machine_%d_Cluster_%d.asv'%ii,i
        filename2="Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) +".asv"
        os.chdir(training_master)
        model0= pickle.load(open(filename2, 'rb'))
        os.chdir(oldfolder)
        labelDA0=(np.asarray(np.where(labelDA == i))).T
#    
#    
#    ##----------------------##------------------------##
        a00=inputtest[labelDA0,:]
        a00=np.reshape(a00,(-1,numcols),'F')
        if a00.shape[0]!=0:
            clementanswer[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))
    clementanswer=scalerz.inverse_transform(clementanswer)
    return clementanswer	
		
#------------------Begin Code-------------------------------------------------------------------#
print('')
print('-------------------LOAD INPUT DATA-------------------------------------')
datafind =  os.path.join(oldfolder,"Data")
Machine_true = "ML_MACHINE"
if os.path.isdir(Machine_true): 
    shutil.rmtree(Machine_true)      
os.mkdir(Machine_true)
np.random.seed(5)
training_master =  os.path.join(oldfolder,Machine_true)
os.chdir(datafind)
inpuutx,outpuutx,df=Read_data_csv("Dataset.csv",training_master,oldfolder,\
                                      "Data")
os.chdir(oldfolder)
inpuutx =inpuutx.astype('float32')
outpuutx =np.reshape(outpuutx.astype('float32'),(-1,1),'F')
raw_signal=outpuutx
sensorr=raw_signal
intee_raw=inpuutx
print('-------------MODEL FITTING FOR INPUT TO OUTPUT RELATIONSHIPS-----------')
print('Using CCR for model fitting for Model :7 to 1')
print('')
print('References for CCR include: ')
print(' (1): David E. Bernholdt, Mark R. Cianciosa, David L. Green, Jin M. Park,\
Kody J. H. Law, and Clement Etienam. Cluster, classify, regress:A general\
method for learning discontinuous functions.Foundations of Data Science, \
1(2639-8001-2019-4-491):491, 2019.')
print('')
print('(2): Clement Etienam, Kody Law, Sara Wade. Ultra-fast Deep Mixtures of\
Gaussian Process Experts. arXiv preprint arXiv:2006.13309, 2020.')
     
print('-----------------------------------------------------------------------')          
outpuutx2=outpuutx
inpuutx2=inpuutx
iniguess=inpuutx2
scaler1a = MinMaxScaler(feature_range=(0, 1))
(scaler1a.fit(outpuutx2))
os.chdir(training_master)
pickle.dump(scaler1a, open("clfy.asv", 'wb'))
os.chdir(oldfolder)
outpuutx2=(scaler1a.transform(outpuutx2))


scaler2a = MinMaxScaler(feature_range=(0, 1))
#inpuutx2=gaussianizeit(inpuutx2)
(scaler2a.fit(inpuutx2))
os.chdir(training_master)
pickle.dump(scaler2a, open("clfx.asv", 'wb'))
os.chdir(oldfolder)
inpuutx2=(scaler2a.transform(inpuutx2))
# inpuut2, X_test2, outpuut2, y_test2 = train_test_split\
# (inpuutx2, outpuutx2, test_size=0.05)


inpuut2=inpuutx2
X_test2=inpuutx2
outpuut2=outpuutx2
y_test2= outpuutx2


def startit(i,outpuut2,inpuut2,training_master,oldfolder,raw_signal):
#    for i in range(sizee):
    print('')
    print('Starting training machine %d'%(i+1))
    useeo=outpuut2[:,i]
    useeo=np.reshape(useeo,(-1,1),'F')

    usein=inpuut2
    usein=np.reshape(usein,(-1,7),'F')                 
        
    clust=CCR_Machine(usein,useeo,i,training_master,oldfolder,raw_signal)

    bigs=clust
    return bigs
    print('')
    print('Finished training machine %d'%(i+1))

inputs = range(1) 
num_cores = multiprocessing.cpu_count()
bigs = Parallel(n_jobs=num_cores, verbose=0)(delayed(
    startit)(i,outpuut2,inpuut2,training_master,oldfolder,raw_signal)for i in inputs)
big = np.vstack(bigs)
os.chdir(training_master)
for i in range (1):
    a=open("clustersizes.dat", "a+")
    a.write("%d \n" % (big[i,:]))
    a.close()
clfy= pickle.load(open("clfy.asv", 'rb'))
clfx= pickle.load(open("clfx.asv", 'rb'))
os.chdir(oldfolder)



def endit(i,clfy,clfx,testt,training_master,oldfolder):
    #for i in range(sizee):
    print('')
    print('Starting prediction from machine %d'%(i+1))
      
    numcols = len(testt[0])
    clemzz=PREDICTION_CCR__MACHINE(i,int(big[i,:]),testt,numcols,clfy,training_master,oldfolder)
    
    print('')
    print('Finished Prediction from machine %d'%(i+1)) 
    return clemzz
#endit()    
#outputreq[:,i]=np.ravel(clemzz)
inputs = range(1) 
num_cores = multiprocessing.cpu_count()
cleme = Parallel(n_jobs=num_cores, verbose=50)(delayed(
    endit)(i,clfy,clfx,X_test2,training_master,oldfolder)for i in inputs)
outputreq = np.hstack(cleme)

outputpred = outputreq
CoDoverallDNN,L_2overallDNN,CoDviewDNN,L_2viewDNN = Performance_plot\
(outputpred,clfy.inverse_transform(y_test2),"Machine_perform",training_master,\
 oldfolder)    
print ('R2 of fit using the DNN machine for model is :', CoDoverallDNN)
print ('L2 of fit using the DNN machine for model is :', L_2overallDNN)

print('---------------------------------------------------------------------')

print('--------------TIME SERRIES MODEL FITTING FOR STATES-------------------')
      
usethis2=pd.date_range(start='2010-01-01 00:00:00', end='2019-12-31 23:45:00', freq="15min")          
datetime_series = pd.Series(usethis2)
spitts=['Ambient temperature (C)','Ground temperature (C)','Global irradiance (W/m2)',\
        'Direct irradiance (W/m2)','Diffuse irradiance (W/m2)']
   
train_size = int(len(outpuutx) * 0.5)

inputs = range(5) 
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores, verbose=50)(delayed(
    Time_series_learning)(df,spitts[i],datetime_series,i,\
                         training_master,oldfolder,train_size)for i in inputs)
                          
print('-------------------PROGRAM EXECUTED-------------------------------------')    