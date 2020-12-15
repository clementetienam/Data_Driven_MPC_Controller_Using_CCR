# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:27:55 2020

@author: Dr Clement Etienam
Time series=XGboost
Forward problem= CCR: XGboost is gates and Polynomial regression are experts

The controler options are : LBFGS,I-ES,ES-MDA,EnKF. Bounded simplicial homology global optimization
"""
from __future__ import print_function
print(__doc__)
import numpy as np
import pickle
from scipy.stats import rankdata, norm
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
import datetime 
import multiprocessing
import os
np.random.seed(5)
from numpy import linalg as LA
import numpy
import scipy.optimize as opt
from copy import copy
from joblib import Parallel, delayed
import scipy
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from functools import partial
from hyperopt import hp,fmin, STATUS_OK
from hyperopt import tpe
import theano
import theano.tensor as tt
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

import random

random.seed(1)
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
    plt.title('Air Temperature [C](10 Minutes)', fontsize = 20)
    plt.ylabel('Machine',fontsize = 20)
    plt.xlabel('True data',fontsize = 20)
    a,b=best_fit(np.ravel(clementanswer2), np.ravel(outputtest2),)
    yfit = [a + b * xi for xi in np.ravel(clementanswer2)]
    plt.plot(np.ravel(clementanswer2), yfit,color='r')
    plt.annotate('R2= %.3f' % CoD2, (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=10)    
    os.chdir(training_master)        
    plt.savefig("%s.jpg"%stringg)
    os.chdir(oldfolder)
    #plt.show()
	
    return CoDoverall,R2overall,CoDview,R2view  
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
    data1=data1.values
    temp=data1
    outpuut=temp[:,[7]]
    inpuut=temp[:,[0,1,2,3,4,5,6]]
    setpoint=temp[:,[8]]
    return inpuut,outpuut ,df,setpoint
	
def PREDICTION_CCR__MACHINE(ii,nclusters,inputtest,numcols,scalerz,training_master,oldfolder):
    import numpy as np
    import pickle
    #from keras.models import load_model
    filename1='Classifier_%d.asv'%ii
    #filename1='Classifier_%d.h5'%ii
    os.chdir(training_master)
    loaded_model = pickle.load(open(filename1, 'rb'))
    #loaded_model = load_model(filename1)

    os.chdir(oldfolder)
    labelDA = loaded_model.predict(inputtest)
    #labelDA = loaded_model.predict(inputtest)
    #labelDA=np.argmax(labelDA, axis=-1) 
    numrowstest=len(inputtest)

    clementanswer=np.zeros((numrowstest,1))
    for i in range(nclusters):
        filename2="Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) +".asv"
        os.chdir(training_master)
        model0= pickle.load(open(filename2, 'rb'))
        os.chdir(oldfolder)
        labelDA0=(np.asarray(np.where(labelDA == i))).T   
#    ##----------------------##------------------------##
        a00=inputtest[labelDA0,:]
        a00=np.reshape(a00,(-1,numcols),'F')
        if a00.shape[0]!=0:
            clementanswer[np.ravel(labelDA0),:]=np.reshape(model0.predict(a00),(-1,1))
    clementanswer=scalerz.inverse_transform(clementanswer)
    return clementanswer		

"""
if there is a sensor, then use this as time t and use the LSTM to predict the next!
"""
def costFunc1(Xn,ytruee,clfx,clfy,theshape,Xbefore,nclusters):
    #X=initial_theta
    import numpy as np
    Xbefore=np.reshape(Xbefore,(1,-1),'F')
    Xn=np.reshape(Xn,(1,-1),'F')
    
    X=np.concatenate((Xbefore,Xn), axis=1)
    X=np.reshape(X,(1,-1),'F')
    Xt=(clfx.transform(X)) 

    yt=PREDICTION_CCR__MACHINE(0,nclusters,Xt,7,clfy,training_master,oldfolder)
	
    #yt = scalera.inverse_transform(np.reshape(machine.predict(Xt),(-1,1),'F'))
    yt = np.reshape(yt, (-1, 1))
    ytruee = np.reshape(ytruee, (-1, 1))
    cosst=(np.sum((ytruee-yt)**2))/2 #+amount +#
    return cosst


def optima_clement1(i,oldfolder,training_master,ytrue,\
                    theshape,iniclemz,Nop,iniclemb,clfx,clfy,nclus):
    import numpy as np
    # os.chdir(training_master)
    # scalera= pickle.load(open("outputtransform1.asv", 'rb'))
    # scaler= pickle.load(open("inputtransform1.asv", 'rb'))
    # #machine = pickle.load(open("Machine.asv", 'rb'))
    # os.chdir(oldfolder)
    print('%d|%d'%((i+1),Nop))
    yuse=np.reshape(ytrue[i,:],(1,-1) ,'F')
    #initial_theta = np.reshape(iniclemz,(1,-1),'F')
    initial_theta=np.reshape(iniclemz[i,:],(1,-1),'F')
    iniclem=np.reshape(iniclemb[i,:],(1,-1),'F')
    resultt = opt.fmin_powell(func=costFunc1, x0=initial_theta,xtol=1e-10,ftol=1e-10, \
                      args=(yuse,clfx,clfy,theshape,iniclem,nclus))
    Xe=np.reshape(resultt,(1,-1),'F')
    # del machine
    # del scaler
    # del scalera
    return Xe 


def optima_clement1a(i,oldfolder,training_master,ytrue,\
                    theshape,iniclemz,Nop,iniclemb,clfx,clfy,nclus,minnss,maxss):
    import numpy as np
    print('%d|%d'%((i+1),Nop))
    yuse=np.reshape(ytrue[i,:],(1,-1) ,'F')
    #initial_theta = np.reshape(iniclemz,(1,-1),'F')
    initial_theta=np.reshape(iniclemz[i,:],(1,-1),'F')
    iniclem=np.reshape(iniclemb[i,:],(1,-1),'F')
    bnds=[(np.asscalar(minnss[:,0]), np.asscalar(maxss[:,0])), (np.asscalar(minnss[:,1]), np.asscalar(maxss[:,1]))]    
    resultt = opt.shgo(func=costFunc1,bounds=bnds, \
                      args=(yuse,clfx,clfy,theshape,iniclem,nclus),n=100, iters=5)
    Xe=np.reshape(resultt.x,(1,-1),'F')
    return Xe 


def objective_Bayes(thetac,kk,oldfolder,training_master,ytrue,\
                    theshape,Nop,augg,clfx,clfy,nclus):
    import numpy as np
    print(str(kk+1) + ' out of ' + str(Nop))
    yuse=np.reshape(ytrue[kk,:],(1,-1) ,'F')
    #initial_theta = np.reshape(iniclemz,(1,-1),'F')
    x1 = np.atleast_1d(thetac['x1'])
    x2=np.atleast_1d(thetac['x2'])
    
    thetaa=np.zeros((1,2))
    thetaa[:,0]=x1
    thetaa[:,1]=x2
    
    #thetaa=np.concatenate((x1,x2), axis=1)
    initial_theta=np.reshape(thetaa,(1,-1),'F')
    import numpy as np
    Xbefore=np.reshape(augg[kk,:],(1,-1),'F')
    Xn=np.reshape(initial_theta,(1,-1),'F')    
    X=np.concatenate((Xbefore,Xn), axis=1)
    X=np.reshape(X,(1,-1),'F')
    Xt=(clfx.transform(X)) 
    yt=PREDICTION_CCR__MACHINE(0,nclus,Xt,7,clfy,training_master,oldfolder)	
    #yt = scalera.inverse_transform(np.reshape(machine.predict(Xt),(-1,1),'F'))
    yt = np.reshape(yuse, (-1, 1))
    ytruee = np.reshape(ytrue, (-1, 1))
    cosst=(np.sum((ytruee-yt)**2))/2 #+amount +#    
    return {'loss': cosst ,  'status': STATUS_OK}
    


def pinvmat(A,tol = 0):
    V,S1,U = np.linalg.svd(A,full_matrices=0)

    # Calculate the default value for tolerance if no tolerance is specified
    if tol == 0:
        tol = np.amax((A.size)*np.spacing(np.linalg.norm(S1,np.inf)))  
    
    r1 = sum(S1 > tol)+1
    v = V[:,:r1-1]
    U1 = U.T
    u = U1[:,:r1-1]
    S11 = S1[:r1-1]
    s = S11[:]
    S = 1/s[:]
    X = (u*S).dot(v.T)

    return X

def create_features2(df,datetime_series):
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
    
    return X
def Time_series_prediction(X_in,ii,\
                         training_master,oldfolder,spaa):


    filename="series" + str(ii) +".asv"
    print('Predicting for'+ spaa)
    os.chdir(training_master)
    reguse= pickle.load(open(filename, 'rb'))
    os.chdir(oldfolder)    
    aa= reguse.predict(X_in)
    return aa


def trunc_gauss(mu, sigma, bottom, top,mu1,sigma1,bottom1,top1):
    a = random.gauss(mu,sigma)
    while (bottom <= a <= top) == False:
        a = random.gauss(mu,sigma)
        
    a1 = random.gauss(mu1,sigma1)
    while (bottom1 <= a1 <= top1) == False:
        a1 = random.gauss(mu1,sigma1)        
        
    ouut=np.zeros((1,2))
    ouut[:,0]=a
    ouut[:,1]=a1   
        
    return ouut



def Get_ensemble2(mu1,sigma1,mu2,sigma2):
    s1 = np.random.normal(mu1, sigma1, 1)
    s2 = np.random.normal(mu2, sigma2, 1)
    ouut=np.zeros((1,2))
    ouut[:,0]=s1
    ouut[:,1]=s2
    return ouut

from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



def funcGetDataMismatch(simData,measurment):

    ne=simData.shape[1]

    objReal=np.zeros((ne,1))
    for j in range(ne):
        objReal[j]=(np.sum((((simData[:,j]) - measurment) ** 2)) )  **(0.5)
    obj=np.mean(objReal)

    objStd=np.std(objReal)
    return obj,objStd,objReal



import pymc3 as pm
def new_mcmc_forwarding(x3,x4,kk,oldfolder,training_master,\
    theshape,Nop,augg,clfx,clfy,nclus):

    kk=0
    import scipy.optimize as optimize
    print(str(kk+1) + ' out of ' + str(Nop))
    yuse=np.reshape(True_signal[kk,:],(1,-1) ,'F')
    #initial_theta = np.reshape(iniclemz,(1,-1),'F')
    theta=np.zeros((1,2))
    # x1=np.asarray(x1,dtype=float)
    # x1=np.atleast_1d(x1)
    # x1=np.reshape(x1,(1,1))
    # x2=np.reshape(x2,(1,1))
    # x2=np.atleast_1d(x2)
    # x2=np.asarray(x2,dtype=float)
    # theta=np.concatenate((x1,x2), axis=1)
    theta=x3+x4
    initial_theta=np.reshape(theta,(1,-1),'F')
    
    import numpy
    Xbefore=numpy.reshape(augg[kk,:],(1,-1),'F')
    Xn=np.reshape(initial_theta,(1,-1),'F')    
    X=np.concatenate((Xbefore,Xn), axis=1)
    X=np.reshape(X,(1,-1),'F')
    X=X.eval()
    Xt=(clfx.transform(X)) 
    yt=PREDICTION_CCR__MACHINE(0,nclus,Xt,7,clfy,training_master,oldfolder)	
    #yt = scalera.inverse_transform(np.reshape(machine.predict(Xt),(-1,1),'F'))
    yt = numpy.reshape(yuse, (-1, 1))    
    return yt
    

def mcmc_optimise(kk,oldfolder,training_master,\
        theshape,Nop,augg,clfx,clfy,nclus,True_signal,minnss,maxss):
    
    basic_model = pm.Model()
    mean1=(np.asscalar(minnss[:,0])+ numpy.asscalar(maxss[:,0]))/2
    mean2=(np.asscalar(minnss[:,1])+ numpy.asscalar(maxss[:,1]))/2
    # mu=np.zeros((1,2))
    # mu[:,0]=mean1
    # mu[:,1]=mean2
    # true_cov = np.array([[0.1*mean1, 0], [0, 0.1*mean2]])
    with basic_model:
        #x1 = pm.Uniform('x1',)  # Prior for x1
        x1=pm.Normal('x1', mu=mean1, sigma=0.1*mean1)
        x2=pm.Normal('x2', mu=mean2, sigma=0.1*mean2)
        # x = pm.MvNormal('x', mu=mu, cov=true_cov,shape=(5, 2))
        
        
       # x2 = pm.Uniform('x2', numpy.asscalar(minnss[:,1]),numpy.asscalar(maxss[:,1]))    # prior for x2
    
        y_pred = pm.Normal('y_pred', yt=new_mcmc_forwarding(x1,x2,kk,oldfolder,training_master,\
        theshape,Nop,augg,clfx,clfy,nclus), observed=True_signal[kk,:])  # bringing it all together
        #trace_g = pm.sample(2000, tune=1000,cores=2)  
        
    map_estimate = pm.find_MAP(model=basic_model,method="powell")
    z1=map_estimate['x1']
    z2=map_estimate['x2']
    clementanswer=np.zeros((1,2))
    clementanswer[:,0]=z1
    clementanswer[:,1]=z2
    return clementanswer



def yess(ij,Nob,minnss,maxss,oldfolder,training_master,True_signal,\
    valuee,Nop,statesj2,clfx,clfy,nclus):
    spacee = {
    'x1': hp.uniform('x1',minnss[:,0],maxss[:,0]),
    'x2': hp.uniform('x2',minnss[:,1],maxss[:,1]),
    'name': hp.choice('name', ['Heat pump heat supply (kW)'
                               'Heat pump electrical load (kW)']),
    }
    
# kk,oldfolder,training_master,ytrue,\
#                     theshape,Nop,augg,clfx,clfy,nclus    
        
    fmin_objective = partial(objective_Bayes, kk=ij,oldfolder=oldfolder,training_master=training_master,ytrue=True_signal,\
    theshape=valuee,Nop=Nop,augg=statesj2,clfx=clfx,clfy=clfy,nclus=nclus)
    best=fmin(fn = fmin_objective ,space = spacee,algo=tpe.suggest, max_evals=Nob)
    x1=best['x1']
    x2=best['x2']
    clemanswer=np.zeros((1,2))
    clemanswer[:,0]=x1
    clemanswer[:,1]=x2
    return clemanswer  

def Forwarding(i,clfx,clfy,aav,statesj2,training_master,oldfolder,nclusters):
    aa=aav[:,i]
    aa=np.reshape(aa,(-1,2),'F')
    aa=np.concatenate((statesj2, aa), axis=1)
    X=np.reshape(aa,(-1,7),'F')
    Xt=(clfx.transform(X)) 
    yt=PREDICTION_CCR__MACHINE(0,nclusters,Xt,7,clfy,training_master,oldfolder)	
    yt = np.reshape(yt,(-1,1),'F')
    yt = np.reshape(yt, (-1, 1))
    return yt
def iES(maxouter,maxiner,y_train,iniensemble,clfx,clfy,suni,statesj2,training_master,oldfolder,nclusters):
    

    #--  initialization  --#
    beta = 0 # $beta$ determines the threshold value in one of the stopping criteria
    maxOuterIter = maxouter; # maximum iteration number in the outer loop
    maxInnerIter = maxiner; # maximum iteration number in the inner loop
    init_lambda = 1; # initial lambda value
    lambda_reduction_factor = 0.9; # reduction factor in case to reduce gamma
    lambda_increment_factor = 2 # increment factor in case to increase gamma 
    doTSVD = 1; # do a TSVD on the cov of simulated obs? (1 = yes)

    tsvdCut = 0.99; # discard eigenvalues/eigenvectors if they are not among the truncated leading ones

    min_RN_change = 2; # minimum residual norm (RN) change (in percentage); RN(k) - RN(k+1) > RN(k) * min_RN_change / 100
    
    iterr=0
    lambda_=init_lambda

    
    # ne=Ne
    # measurement=True_signal
    # y_train=True_signal
    # iniensemble=ini_ensemble
    # clfx=scalerinput1
    # clfy=scaleroutput1
    # model=Dnn1_machine

    measurement=copy(y_train)
    nd=len(measurement)
   
    ensemble=(iniensemble)
    ne=ensemble.shape[1]

    meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
    
    ensemble=np.concatenate((ensemble, meann), axis=1)

    perturbedData=np.zeros((nd,ne))


    W=np.ones(len(measurement))
    W=np.reshape(W,(-1,1))


    for j in range(ne):
        s = np.random.normal(0, 1, len(measurement))
        s=np.reshape(s,(-1,1))
        perturbedData[:,j]=np.ravel(measurement + np.multiply(W,s))

    
    # simulated observations of the ensemble members
    nm=len(ensemble)
    simData=np.zeros((nd,ensemble.shape[1]))

        
    Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
    Forwarding)(i,clfx,clfy,ensemble,statesj2,training_master,oldfolder,nclusters)for i in range(ne+1) )

    simData = np.reshape(np.hstack(Xspit),(nd,ne+1),'F')# Thease are the predicted  states        

    #del(aa)
    # data mismatch w.r.t the ensemble
    obj,objStd,objReal=funcGetDataMismatch(simData[:,:ne],measurement)
    init_obj=(obj)
    init_objStd=(objStd)
    # load information from methodInfo to configure iES
    objThreshold=np.dot(beta ** 2,nd)

    isTooSmallRNChange=0

    exitFlag=np.array([0,0,0])

   
    #-------------#
    #-- run iES --#
    #-------------#
    #tsvdCut=0.99
    # outer iteration loops
    while (iterr < maxOuterIter) and  (obj > objThreshold):
        #print(str(iterr+1) + '  out of ' + str(maxOuterIter))
        
        deltaM=np.ones((nm,ne))
        deltaD=np.ones((nd,ne))
		
        for i in range(ne):
		
            deltaM[:,i]=ensemble[:,i]- ensemble[:,-1]
            # deviation of the simulated observations
            deltaD[:,i]=simData[:,i] - simData[:,-1]
        Ud,Wd,Vd= np.linalg.svd(deltaD, full_matrices=True)
       # Wd=np.reshape(Wd,(-1,1))
        val=np.diag(Wd)
        total=np.sum(val)
        for j in range(ne):
            svdPd=copy(j)
            if (np.sum(val[:j]) / total > tsvdCut):
            
                break
        #svdPd=np.asscalar(svdPd)    
        Vd=Vd[:,:svdPd]
        Ud=Ud[:,:svdPd]
        Wd=Wd[:svdPd]
        del(val)
        #-- initialization of the inner loop --#
        iterLambda=1
        while iterLambda < maxInnerIter:
            ensembleOld=(ensemble)
            simDataOld=(simData)
            if doTSVD==1:
                alpha=np.dot(lambda_,np.sum(Wd ** 2)) / svdPd
                yy=Wd / (Wd ** 2 + alpha)
                yy=np.reshape(yy,(-1,1))
                x1=np.dot(Vd,scipy.sparse.spdiags(yy.T,0,svdPd,svdPd).toarray())
                KGain=np.dot(np.dot(deltaM,x1),Ud.T)

                
            else:
                alpha=np.dot(lambda_,np.sum(np.sum(deltaD ** 2))) / nd
                
                first=np.dot(deltaM,deltaD.T)
                second=(np.dot(deltaD,deltaD.T) + np.dot(alpha,np.eye(nd)))
                
                Usig,Sig,Vsig = np.linalg.svd(second, full_matrices = True)
                Bsig = np.cumsum(Sig, axis = 0)          # vertically addition
                valuesig = Bsig[-1]                 # last element
                valuesig = valuesig * 0.99
                indices = ( Bsig >= valuesig ).ravel().nonzero()
                toluse = Sig[indices]
                tol = toluse[0]
                
                KGain = np.dot(first,pinvmat(second,tol))
                #KGain=np.dot(first,np.linalg.pinv(second))
            simData=np.zeros((nd,ensemble.shape[1]))   
            iterated_ensemble=ensemble[:,:ne]- np.dot(KGain,(simData[:,:ne]- perturbedData))
            meann=np.reshape(np.mean(iterated_ensemble,axis=1),(-1,1),'F')
            ensemble=np.concatenate((iterated_ensemble, meann), axis=1)
            changeM=np.sqrt(np.sum((ensemble[:,ne] - ensembleOld[:,ne]) ** 2) / nm)
            simData=np.zeros((nd,ne+1))

           # del(simData)
            Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Forwarding)(i,clfx,clfy,ensemble,statesj2,training_master,oldfolder,nclusters)for i in range(ne+1) )
            simData = np.reshape(np.hstack(Xspit),(nd,ne+1),'F')# Thease are the predicted  states                        
            objNew,objStdNew,objRealNew=funcGetDataMismatch(simData[:,:ne],measurement)
            tmp_objReal=(objReal)
            objReal=(objRealNew)
            objReal=(tmp_objReal)
            del(tmp_objReal)
            if objNew > obj:
                lambda_=np.dot(lambda_,lambda_increment_factor)
                iterLambda=iterLambda+ 1
                simData=(simDataOld)
                ensemble=(ensembleOld)
            else:
                changeStd=(objStdNew - objStd) / objStd
                lambda_=np.dot(lambda_,lambda_reduction_factor)
                iterr=iterr+1
                simDataOld=(simData)
                ensembleOld=(ensemble)
                objStd=(objStdNew)
                obj=(objNew)
                objReal=(objRealNew)
                break

        # if a better update not successfully found
        if iterLambda >= maxInnerIter:
            lambda_=np.dot(lambda_,lambda_increment_factor)
            if lambda_ < init_lambda:
                lambda_=init_lambda
            iterr=iterr+1
        if isTooSmallRNChange:

            exitFlag[3]=1
            break

    
    if iterr >= maxOuterIter:
        exitFlag[1]=1

    if obj <= objThreshold:

        exitFlag[2]=1   
    return ensemble,exitFlag


def ES_MDA(num_ens,m_ens,Z,prod_ens,alpha,CD,numsave=2):
    varn=0.99#1-1/math.pow(10,numsave)
    # Initial Variavel 
    # Forecast step
    yf = m_ens                        # Non linear forward model 
    df = prod_ens                     # Observation Model
    numsave
    ym = np.array(yf.mean(axis=1))    # Mean of the y_f
    dm = np.array(df.mean(axis=1))    # Mean of the d_f
    ym=ym.reshape(ym.shape[0],1)    
    dm=dm.reshape(dm.shape[0],1)    
    dmf = yf - ym
    ddf = df - dm
    
    Cmd_f = (np.dot(dmf,ddf.T))/(num_ens-1);  # The cros-covariance matrix
    Cdd_f = (np.dot(ddf,ddf.T))/(num_ens-1);  # The auto covariance of predicted data
    
    # Perturb the vector of observations
    R = scipy.linalg.cholesky(CD,lower=True) #Matriz triangular inferior
    U = R.T   #Matriz R transposta
    p , w =scipy.linalg.eig(CD)
    
    aux = np.repeat(Z,num_ens,axis=1)
    mean = 0*(Z.T)

    noise=np.random.multivariate_normal(mean[0], np.eye(len(Z)), num_ens).T
    d_obs = aux+math.sqrt(alpha)*np.dot(U,noise)  
    
    # Analysis step
    u, s, vh = scipy.linalg.svd(Cdd_f+alpha*CD); v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ss = np.diag(diagonal**(-1))
    K=np.dot(Cmd_f,(np.dot(np.dot(v,ss),(u.T))))
    # Use Kalman covariance
    # if len(corr)>0:
    #     K = corr*K
        
    ya = yf + (np.dot(K,(d_obs-df)))
    m_ens = ya
    return m_ens

def center(E, axis=0, rescale=False):
    """Center ensemble.

    Makes use of np features: keepdims and broadcasting.

    - rescale: Inflate to compensate for reduction in the expected variance."""
    x = np.mean(E, axis=axis, keepdims=True)
    X = E - x

    if rescale:
        N = E.shape[axis]
        X *= np.sqrt(N/(N-1))

    x = x.squeeze()

    return X, x
def mean0(E, axis=0, rescale=True):
    "Same as: center(E,rescale=True)[0]"
    return center(E, axis=axis, rescale=rescale)[0]


def inflate_ens(E, factor):
    "Inflate the ensemble (center, inflate, re-combine)."
    if factor == 1:
        return E
    X, x = center(E)
    return x + X*factor


def mrdiv(b, A):
    import scipy.linalg as sla
    "b/A"
    return sla.solve(A.T, b.T).T


def mldiv(A, b):
    import scipy.linalg as sla
    "A \\ b"
    return sla.solve(A, b)

def EnKF(ensemble, Sim, CDd, Ne,True_single):
    
    #Sim=simDatafinal
    R=CDd
    N=Ne
    ttt=True_single.T
    Ef=ensemble.T
    hE=Sim.T
    """ANALYSIS STEP"""  
    from numpy.random import multivariate_normal
    mu = mean0(Ef,0) #ensemble mean
    A  = Ef - mu #enesemble anomaly
    hx = mean0(hE,0)
    Y  = hE-hx #anomalies of the observed ensemble
    D = multivariate_normal([0]*len(ttt.T), R, N) #https://github.com/rlabbe/Kalman-anYd-Bayesian-Filters-in-Python/blob/master/Appendix-E-Ensemble-Kalman-Filters.ipynb
    #D=D.T
    C  =  Y.T @ Y +( R*(N-1))
    YC =  mrdiv(Y, C)
    KG = A.T @ YC         
    dE = (KG @ ( ttt + D - hE ).T).T 
    Ea  = Ef + dE 
    clemt=Ea.T
    return clemt

#------------------Begin Code-------------------------------------------------------------------#
print('')
print('-------------------LOAD INPUT DATA-------------------------------------')
datafind =  os.path.join(oldfolder,"Data")
Machine_true = "ML_MACHINE"
training_master =  os.path.join(oldfolder,Machine_true)
Nop=int(input('Enter the sample size to test controller: '))

os.chdir(datafind)
inpuutx,outpuutx,df,setpoint=Read_data_csv("Dataset.csv",training_master,oldfolder,\
                                      "Data")
os.chdir(oldfolder)
inpuutx =inpuutx.astype('float32')
outpuutx =np.reshape(outpuutx.astype('float32'),(-1,1),'F')
setpoint =np.reshape(setpoint.astype('float32'),(-1,1),'F')
      
usethis2=pd.date_range(start='2010-01-01 00:00:00', end='2019-12-31 23:45:00', freq="15min")          
datetime_series = pd.Series(usethis2)

tola = datetime_series  
tola = pd.DataFrame(tola)  
periodss = tola.shape[0]

spitts=['Ambient temperature (C)','Ground temperature (C)','Global irradiance (W/m2)',\
        'Direct irradiance (W/m2)','Diffuse irradiance (W/m2)']
    

train_size = int(len(outpuutx) * 0.5)


    
future = [None] * periodss 
future = pd.DataFrame(future)
future.columns = ['Date/Time']
future['Date/Time'] = datetime_series
Xfuture=create_features2(future,datetime_series) 
Xreq=Xfuture.iloc[train_size:train_size+Nop,:]
tolareq=tola.iloc[train_size:train_size+Nop,:]
True_signal=setpoint [train_size:train_size+Nop,:]



print('')
print ('-----------------------load the Machines-----------------------------')
os.chdir(training_master)
clfy= pickle.load(open("clfy.asv", 'rb'))
clfx= pickle.load(open("clfx.asv", 'rb'))
print(' Determine how many clusters were used for training CCR')
cluster_all = np.genfromtxt("clustersizes.dat", dtype='float')
cluster_all=np.reshape(cluster_all,(-1,1),'F')
nclus=int(cluster_all[0,:])
os.chdir(oldfolder)

inputs = range(5) 
num_cores = multiprocessing.cpu_count()
Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=50)(delayed(
    Time_series_prediction)(Xreq,i,\
                         training_master,oldfolder,spitts[i])for i in inputs)

statesj2 = np.reshape(np.hstack(Xspit),(Nop,5),'F')# Thease are the predicted  states

sampl3=np.zeros((Nop,5))
iniguess2=inpuutx[:,[0,1,2,3,4]]
minnss=np.reshape(np.min(iniguess2, axis=0),(-1,5),'F')
maxss=np.reshape(np.max(iniguess2, axis=0),(-1,5),'F')


for jj in range(5):
    aa=np.reshape(statesj2[:,jj],(-1,1),'F')
    minsc=np.asscalar(minnss[:,jj])
    maxcc=np.asscalar(maxss[:,jj])
    
    # clemlow=np.atleast_1d(np.reshape(np.repeat(minsc, Nop),(-1,1),'F'))
    # clemhigh=np.reshape(np.repeat(maxcc, Nop),(-1,1),'F')
    
    aa[aa<=minsc]=minsc
    
    aa[aa>=maxcc]=maxcc
    sampl3[:,jj]=np.ravel(np.reshape(aa,(-1,1),'F'))
    
statesj2=sampl3
print('')

print('-----------------------Optimise the control----------------------------')
sampl2=np.zeros((Nop,2))
iniguess2=inpuutx[:,[5,6]]
minnss=np.reshape(np.min(iniguess2, axis=0),(-1,2),'F')
maxss=np.reshape(np.max(iniguess2, axis=0),(-1,2),'F')
for j in range(2):
    aa = np.random.uniform(low=minnss[:,j], high=maxss[:,j], size=(Nop,1))
    aa=np.reshape(aa,(Nop,),'F')
    sampl2[:,j]=aa
    
inputss = range(Nop) 
theshape2=5

print('Technique MCMC takes longer and is discouraged')
Technique=int(input('Enter optimisation: 1-LBFGS, 2=IES 3=ESMDA, 4=EnKF, 5=Diffrential_evolution: '))



if Technique==1:
    print('LBFGS chosen')
    method=int(input('Enter method for LBFGS, 1=parallel, 2=sequential: '))
elif Technique==2:
    print('IES chosen)')
    
    methodIES=int(input('Enter method for I-es, 1=Bulk, 2=sequential: '))
    
    if methodIES==1:
        
        Ne=500
        maxouter=40
        maxiner=25
        
    else:
        
        #maxouter=int(input('maximum iteration number in the outer loop (20-40): '))
        maxouter=40
        #maxiner= int(input('maximum iteration number in the inner loop (5-25)): '))
        maxiner=25
        #Ne=int(input('Enter the ensemble size (100-200): '))
        Ne=200
elif Technique==3:
    print('ESMDA chosen')        
    #Ne=100
    alpha=100
    methodESMDA=int(input('Enter method for ESMDA, 1=Bulk, 2=sequential: '))
    if methodESMDA==1:
        Ne=500
    else:
        Ne=100
elif Technique==4:
    print('EnKF chosen')   

    #Ne=100
    alpha=10
    methodEnKF=int(input('Enter method for EnKF, 1=Bulk, 2=sequential: '))
    if methodEnKF==1:
        Ne=500
    else:
        Ne=100

else:
    print('Diffrential Evolution chosen')
    print(' This method gives a bounded optimisation')
    methodDiff=int(input('Enter method , 1=parallel, 2=sequential: '))    
        
if Technique==1:    

    if method==1:
        print('Parallel implementation')
        num_cores = multiprocessing.cpu_count()
        Xrequired2 = Parallel(n_jobs=25, backend='loky',verbose=50)(delayed(
            optima_clement1)(i,oldfolder,training_master,True_signal,\
                           theshape2,sampl2,Nop,statesj2,clfx,clfy,nclus)\
                for i in inputss)
        controlj2 = np.vstack(Xrequired2)
    else:    
    #if method==2:
        print('Series implementation')
        Xrequired3=np.zeros((Nop,2))
        
        for i in range(Nop):
            Xrequired3[i,:]=(
            optima_clement1)(i,oldfolder,training_master,True_signal,\
                           theshape2,sampl2,Nop,statesj2,clfx,\
                               clfy,nclus,minnss,maxss)
        controlj2=Xrequired3
elif Technique==2:
    print('I-ES')
    print('The implementation here follows the paper "Levenberg–Marquardt forms of the \
    iterative ensemble smoother for efficient history matching and uncertainty \
    quantification" by Chen and Oliver, Computational Geosciences, 2013. \
    By Dr Clement Etienam, PhD Petroleum Engineering 2018')
       
    if methodIES==1:
        print('Bulk I-ES')
        
        Truee=True_signal
        statesuse=statesj2
        ini_ensemble=np.zeros((Nop*2,Ne))
        sampl2=np.zeros((Nop,2))
        mezz1=(minnss[:,0]+maxss[:,0])/2
        mezz2=(minnss[:,1]+maxss[:,1])/2
        for kk in range(Ne):
            for ii in range(Nop):
                sampl2[ii,:]=trunc_gauss(mezz1, 0.05*mezz1,minnss[:,0], maxss[:,0],mezz2,0.05*mezz2,minnss[:,1],maxss[:,1])
            ini_ensemble[:,kk]=(np.reshape(sampl2,(-1,),'F')) 
        ensembleout,exitFlag=iES(maxouter,maxiner,Truee,ini_ensemble,clfx,\
                clfy,2,statesuse,training_master,oldfolder,nclus)  
        
        controljj2= np.reshape(ensembleout[:,-1] ,(-1,2),'F')
        controlj2=controljj2
        Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
        Forwarding)(i,clfx,clfy,ensembleout,statesuse,training_master,oldfolder,nclus)for i in range(Ne+1) )
        simDatafinal = np.reshape(np.hstack(Xspit),(Nop,Ne+1),'F')# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_signal)
        clem=np.where(cc == cc.min())
        clem=np.asscalar(clem[0])
        controlbest= np.reshape(ensembleout[:,clem] ,(-1,2),'F')
        controlbest2=controlbest    
           
    else:
        
        print('Sequential IES')
        controlj2=np.zeros((Nop,2))
        controlbest2=np.zeros((Nop,2))
    
        
        for i in range(Nop):
            print( str(i+1) + ' out of ' + str(Nop))
            Truee=True_signal[i,:]
            Truee=np.reshape(Truee,(1,-1),'F')
            statesuse=statesj2[i,:]
            statesuse=np.reshape(statesuse,(1,-1),'F')
            sampl2=np.zeros((Nop,2))
            ini_ensemble=np.zeros((1*2,Ne))
            
            mezz1=(minnss[:,0]+maxss[:,0])/2
            mezz2=(minnss[:,1]+maxss[:,1])/2
            for kk in range(Ne):
                
                #sampl2=Get_ensemble2(mezz1,0.2*mezz1,mezz2,0.2*mezz2)
                sampl2=trunc_gauss(mezz1, 0.01*mezz1,minnss[:,0], maxss[:,0],mezz2,0.01*mezz2,minnss[:,1],maxss[:,1])
                ini_ensemble[:,kk]=(np.reshape(sampl2,(-1,),'F'))        
            
            ensembleout,exitFlag=iES(maxouter,maxiner,Truee,ini_ensemble,clfx,\
                            clfy,2,statesuse,training_master,oldfolder,nclus)  
            
            controljj2= np.reshape(ensembleout[:,-1] ,(-1,2),'F')
            controlj2[i,:]=controljj2
            Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Forwarding)(i,clfx,clfy,ensembleout,statesuse,training_master,oldfolder,nclus)for i in range(Ne+1) )
            simDatafinal = np.reshape(np.hstack(Xspit),(1,Ne+1),'F')# 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_signal)
            clem=np.where(cc == cc.min())
            clem=np.asscalar(clem[0])
            controlbest= np.reshape(ensembleout[:,clem] ,(-1,2),'F')
            controlbest2[i,:]=controlbest
               
elif Technique==3:
    
    if methodESMDA==1:
        print('bulk ES-MDA')
        Truee=True_signal
        statesuse=statesj2
        ini_ensemble=np.zeros((Nop*2,Ne))
        sampl2=np.zeros((Nop,2))
        mezz1=(minnss[:,0]+maxss[:,0])/2
        mezz2=(minnss[:,1]+maxss[:,1])/2
        for kk in range(Ne):
            for ii in range(Nop):
                sampl2[ii,:]=trunc_gauss(mezz1, 0.05*mezz1,minnss[:,0], maxss[:,0],mezz2,0.05*mezz2,minnss[:,1],maxss[:,1])
            ini_ensemble[:,kk]=(np.reshape(sampl2,(-1,),'F')) 
            
        ensemble=ini_ensemble
        #ax=np.zeros((Nop,Ne))
        ax=np.zeros((Nop,1))
        for iq in range(Nop):
           # ax=0.1*True_signal
            #ax[iq,:] = np.random.normal(1, 0.1, Ne)
            ax[iq,:]=1
        ax=np.reshape(ax,(-1,))    
        CDd=np.diag(ax)
        #CDd=np.dot(ax,ax.T)
        for ii in range(alpha):
            
            print( str(ii+1) + ' out of ' + str(alpha))
                   
            Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Forwarding)(iy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iy in range(Ne) )
            simDatafinal = np.reshape(np.hstack(Xspit),(Nop,Ne),'F')# Thease are the predicted  states  
            
            updated_ensemble=ES_MDA(Ne,ensemble,Truee,simDatafinal,alpha,CDd,numsave=2)
            ensemble=updated_ensemble
            
        meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
        controljj2= np.reshape(meann,(-1,2),'F')  
        controlj2=controljj2
        
        Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
        Forwarding)(iyy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iyy in range(Ne) )
        simDatafinal = np.reshape(np.hstack(Xspit),(Nop,Ne),'F')# 
        aa,bb,cc=funcGetDataMismatch(simDatafinal,True_signal)
        clem=np.where(cc == cc.min())
        clem=np.asscalar(clem[0])
        controlbest= np.reshape(ensemble[:,clem] ,(-1,2),'F')
        controlbest2=controlbest

    else:
        print('Sequential ES-MDA')
        controlj2=np.zeros((Nop,2))
        controlbest2=np.zeros((Nop,2))
    
        for i in range(Nop):
            print(str(i+1) + ' out of '+ str(Nop))
            Truee=True_signal[i,:]
            Truee=np.reshape(Truee,(1,-1),'F')
            statesuse=statesj2[i,:]
            statesuse=np.reshape(statesuse,(1,-1),'F')
            ini_ensemble=np.zeros((1*2,Ne))
            sampl2=np.zeros((1,2))
            mezz1=(minnss[:,0]+maxss[:,0])/2
            mezz2=(minnss[:,1]+maxss[:,1])/2
            for kk in range(Ne):
                sampl2=trunc_gauss(mezz1, 0.05*mezz1,minnss[:,0], maxss[:,0],mezz2,0.05*mezz2,minnss[:,1],maxss[:,1])
                ini_ensemble[:,kk]=(np.reshape(sampl2,(-1,),'F'))   
    
                    
            ensemble=ini_ensemble
            ax=np.zeros((1,1))
    
            #ax=0.1*True_signal[i,:]
            ax=1
            ax=np.reshape(ax,(-1,))    
            CDd=np.diag(ax)
            
    
            for ii in range(alpha):
                
                #print( str(ii+1) + ' out of ' + str(alpha))
                       
                Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
                Forwarding)(iy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iy in range(Ne) )
                simDatafinal = np.reshape(np.hstack(Xspit),(1,Ne),'F')# Thease are the predicted  states  
                
                updated_ensemble=ES_MDA(Ne,ensemble,Truee,simDatafinal,alpha,CDd,numsave=2)
                ensemble=updated_ensemble
                
            meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
            controljj2= np.reshape(meann,(-1,2),'F')  
            controlj2[i,:]=controljj2
            
            Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Forwarding)(iyy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iyy in range(Ne) )
            simDatafinal = np.reshape(np.hstack(Xspit),(1,Ne),'F')# 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_signal)
            clem=np.where(cc == cc.min())
            clem=np.asscalar(clem[0])
            controlbest= np.reshape(ensemble[:,clem] ,(-1,2),'F')
            controlbest2[i,:]=controlbest

elif Technique==4:
    if methodEnKF==1:
            print('bulk EnKF')
            Truee=True_signal
            statesuse=statesj2
            ini_ensemble=np.zeros((Nop*2,Ne))
            sampl2=np.zeros((Nop,2))
            mezz1=(minnss[:,0]+maxss[:,0])/2
            mezz2=(minnss[:,1]+maxss[:,1])/2
            for kk in range(Ne):
                for ii in range(Nop):
                    sampl2[ii,:]=trunc_gauss(mezz1, 0.05*mezz1,minnss[:,0], maxss[:,0],mezz2,0.05*mezz2,minnss[:,1],maxss[:,1])
                ini_ensemble[:,kk]=(np.reshape(sampl2,(-1,),'F')) 
                
            ensemble=ini_ensemble
            #ax=np.zeros((Nop,Ne))
            ax=np.zeros((Nop,1))
            for iq in range(Nop):
               # ax=0.1*True_signal
                #ax[iq,:] = np.random.normal(1, 0.1, Ne)
                ax[iq,:]=1
            ax=np.reshape(ax,(-1,))    
            CDd=np.diag(ax)
            #CDd=np.dot(ax,ax.T)
            for ii in range(alpha):
                
                print( str(ii+1) + ' out of ' + str(alpha))
                       
                Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
                Forwarding)(iy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iy in range(Ne) )
                simDatafinal = np.reshape(np.hstack(Xspit),(Nop,Ne),'F')# Thease are the predicted  states  
                
                updated_ensemble=EnKF(ensemble, simDatafinal, CDd, Ne,Truee)
                ensemble=updated_ensemble
                
            meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
            controljj2= np.reshape(meann,(-1,2),'F')  
            controlj2=controljj2
            
            Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Forwarding)(iyy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iyy in range(Ne) )
            simDatafinal = np.reshape(np.hstack(Xspit),(Nop,Ne),'F')# 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_signal)
            clem=np.where(cc == cc.min())
            clem=np.asscalar(clem[0])
            controlbest= np.reshape(ensemble[:,clem] ,(-1,2),'F')
            controlbest2=controlbest

    else:
        print('Sequential EnKF')
        controlj2=np.zeros((Nop,2))
        controlbest2=np.zeros((Nop,2))
    
        for i in range(Nop):
            print(str(i+1) + ' out of '+ str(Nop))
            Truee=True_signal[i,:]
            Truee=np.reshape(Truee,(1,-1),'F')
            statesuse=statesj2[i,:]
            statesuse=np.reshape(statesuse,(1,-1),'F')
            ini_ensemble=np.zeros((1*2,Ne))
            sampl2=np.zeros((1,2))
            mezz1=(minnss[:,0]+maxss[:,0])/2
            mezz2=(minnss[:,1]+maxss[:,1])/2
            for kk in range(Ne):
                sampl2=trunc_gauss(mezz1, 0.05*mezz1,minnss[:,0], maxss[:,0],mezz2,0.05*mezz2,minnss[:,1],maxss[:,1])
                ini_ensemble[:,kk]=(np.reshape(sampl2,(-1,),'F'))   
    
                    
            ensemble=ini_ensemble
            ax=np.zeros((1,1))
    
            #ax=0.1*True_signal[i,:]
            ax=1
            ax=np.reshape(ax,(-1,))    
            CDd=np.diag(ax)
            
    
            for ii in range(alpha):
                
                #print( str(ii+1) + ' out of ' + str(alpha))
                       
                Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
                Forwarding)(iy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iy in range(Ne) )
                simDatafinal = np.reshape(np.hstack(Xspit),(1,Ne),'F')# Thease are the predicted  states  
                
                updated_ensemble=EnKF(ensemble, simDatafinal, CDd, Ne,Truee)
                ensemble=updated_ensemble
                
            meann=np.reshape(np.mean(ensemble,axis=1),(-1,1),'F')
            controljj2= np.reshape(meann,(-1,2),'F')  
            controlj2[i,:]=controljj2
            
            Xspit=Parallel(n_jobs=num_cores,backend='loky', verbose=0)(delayed(
            Forwarding)(iyy,clfx,clfy,ensemble,statesuse,training_master,oldfolder,nclus)for iyy in range(Ne) )
            simDatafinal = np.reshape(np.hstack(Xspit),(1,Ne),'F')# 
            aa,bb,cc=funcGetDataMismatch(simDatafinal,True_signal)
            clem=np.where(cc == cc.min())
            clem=np.asscalar(clem[0])
            controlbest= np.reshape(ensemble[:,clem] ,(-1,2),'F')
            controlbest2[i,:]=controlbest  
            
else:
    
    if methodDiff==1:
        print('Parallel implementation')
        num_cores = multiprocessing.cpu_count()
        Xrequired2 = Parallel(n_jobs=30, backend='loky',verbose=50)(delayed(
            optima_clement1a)(i,oldfolder,training_master,True_signal,\
                           theshape2,sampl2,Nop,statesj2,clfx,clfy,nclus,minnss,maxss)\
                for i in inputss)
        controlj2 = np.vstack(Xrequired2)
    else:    
    #if method==2:
        print('Series implementation')
        Xrequired3=np.zeros((Nop,4))
        
        for i in range(Nop):
            Xrequired3[i,:]=(
            optima_clement1a)(i,oldfolder,training_master,True_signal,\
                           theshape2,sampl2,Nop,statesj2,clfx,\
                               clfy,nclus,minnss,maxss)
        controlj2=Xrequired3               
            

print('')

if (Technique == 1) or (Technique==5):
    
    controltony=controlj2

else:
    controltony=controlbest2
    
checkfeedback=np.concatenate((statesj2,controltony), axis=1) 
yycheck=PREDICTION_CCR__MACHINE(0,nclus,clfx.transform(checkfeedback),7,clfy,training_master,oldfolder)	
CoD2=mean_squared_error(True_signal, yycheck)

CoD3=r2_score(True_signal, yycheck)*100
clementanswer2=yycheck
outputtest2=True_signal
outputreq=np.zeros((Nop,1))
for i in range(Nop):
	outputreq[i,:]=outputtest2[i,:]-np.mean(outputtest2)
CoDspa=1-(LA.norm(outputtest2-clementanswer2)/LA.norm(outputreq))
CoD4=1 - (1-CoDspa)**2 ;
CoD4=CoD4*100

print('')
print('Show trend')
stringg2="Summary"
plt.figure(figsize =(20,20))
#plt.subplot(1,2,1)
plt.plot(yycheck[:,0],'ro', color = 'blue', label = 'Controller temperature')
plt.plot(True_signal[:,0],color = 'red', label = 'Set Point temperature')
plt.title('Internal temperature (C) ', fontsize = 20)
plt.ylabel('Temperature [C]',fontsize = 20)
plt.xlabel('Time(15-mins interval)',fontsize = 20)
plt.annotate('Discomfort= %.3fC ' % CoD2, (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=20) 
plt.annotate('R2= %.3f%%' % CoD3, (0.9, 0.3), xycoords='axes fraction', ha='center', va='center', size=20) 
plt.ylim((10, 30)) 	
plt.legend()
os.chdir(training_master)
plt.savefig("%s.jpg"%stringg2)
os.chdir(oldfolder)
plt.show()

print('--------------------SAVE PREDCTION TO FILE----------------------------')


header = '%16s,%16s,%16s,%16s,%16s,%16s,%16s'%('Ambient temperature (C)',\
                                                    'Ground temperature (C)','Global irradiance (W/m2)',\
                                                   'Direct irradiance (W/m2)','Diffuse irradiance (W/m2)',\
                                                    'Heat pump heat supply (kW)','Heat pump electrical load (kW)')
np.savetxt('predicted.CSV',checkfeedback, fmt = '%4.4f',delimiter=',' ,header=header, newline = '\n',comments='')  


header2 = '%16s,%16s'%( 'Heat pump heat supply (kW)','Heat pump electrical load (kW)')
np.savetxt('controller.CSV',controlj2 , fmt = '%4.4f',delimiter=',' ,header=header2, newline = '\n',comments='')  





tolareq.columns=['date']
checkfeedbackultimate=np.concatenate((checkfeedback,yycheck,True_signal), axis=1) 
num_rows, num_cols = checkfeedbackultimate.shape

spittsbig=['Ambient temperature (C)','Ground temperature (C)','Global irradiance (W/m2)',\
        'Direct irradiance (W/m2)','Diffuse irradiance (W/m2)',\
            'Heat pump heat supply (kW)','Heat pump electrical load (kW)',\
                'Attained Temperature(C)','Set Point (C)']

for i in range(num_cols):
    tolareq[spittsbig[i]]=checkfeedbackultimate[:,i]
    
    
#os.chdir(training_master)
tolareq.to_csv('Complete_Prediction.csv',sep=',' )  
#os.chdir(oldfolder) 

print('')       
end = datetime.datetime.now()
timetaken = end - start
print(' Time taken : '+ str(timetaken))
print('-------------------PROGRAM EXECUTED-------------------------------------')    

