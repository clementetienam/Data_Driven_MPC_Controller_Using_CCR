%%
clc;
clear;
close all;
disp('@author: Dr Clement Etienam')
%%{
fprintf('Box Configuration \n')
fprintf('Inputs: \n')
fprintf('Environment:Site Outdoor Air Drybulb Temperature [C] \n')
fprintf('Environment:Site Outdoor Air Wetbulb Temperature [C] \n')
fprintf('Environment:Site Outdoor Air Relative Humidity [%%] \n')
fprintf('Environment:Site Wind Speed [m/s] \n')
fprintf('Environment:Site Wind Direction [deg] \n')
fprintf('Environment:Site Horizontal Infrared Radiation Rate per Area [W/m2] \n')
fprintf('Environment:Site Diffuse Solar Radiation Rate per Area [W/m2] \n')
fprintf('Environment:Site Direct Solar Radiation Rate per Area [W/m2] \n')
fprintf('THERMAL ZONE: BOX:Zone Outdoor Air Wind Speed [m/s] \n')
fprintf('Outputs: \n')
fprintf('THERMAL ZONE: BOX:Zone Mean Air Temperature [C] \n')

fprintf('GSHP configuration \n')
fprintf('Inputs:\n')
fprintf('Environment:Site Outdoor Air Drybulb Temperature [C] \n')
fprintf('Environment:Site Outdoor Air Wetbulb Temperature [C] \n')
fprintf('Environment:Site Outdoor Air Relative Humidity [%%] \n')
fprintf('Environment:Site Wind Speed [m/s] \n')
fprintf('Environment:Site Wind Direction [deg] \n')
fprintf('Environment:Site Horizontal Infrared Radiation Rate per Area [W/m2] \n')
fprintf('Environment:Site Diffuse Solar Radiation Rate per Area [W/m2] \n')
fprintf('Environment:Site Direct Solar Radiation Rate per Area [W/m2] \n')
fprintf('THERMAL ZONE: BOX:Zone Outdoor Air Wind Speed [m/s] \n')
fprintf('GSHPCLG:Heat Pump Electric Power [W] \n')
fprintf('GSHPCLG:Heat Pump Source Side Inlet Temperature [C] \n')
fprintf('GSHPHEATING:Heat Pump Electric Power [W] \n')
fprintf('GSHPHEATING:Heat Pump Source Side Inlet Temperature [C] \n')

fprintf('Outputs: \n')
fprintf('THERMAL ZONE: BOX:Zone Mean Air Temperature [C] \n')

fprintf('Data Driven MPC Approach. Online approach \n')
fprintf('steps: \n')
fprintf('1) Predict room temperature at time t given current weather states \n')
fprintf('2) Optimise for control at time t to reference room temperature \n')
fprintf('3) Predict room temperature at time t+1 using temperature at time t \n')
fprintf('4) Predict weather states for t+1 using temperature at t+1 (gotten from 3) \n')
fprintf('4) Set for next evolution, temperature at t= temperature at t+1 \n')
fprintf( '(prior(t)= posterior(t+1)) \n')
    
fprintf('mathematically; \n')   
fprintf('y=room temperature \n')
fprintf('X=weather states \n')
fprintf('u=control for GSHP pump \n')

fprintf('input: u(t-1)= initial guess,X(t-1)(= Known), r for all t(= known), ... \n')
fprintf('f1, f2,g (= Learned), \n')
fprintf('y(t-1)(=Infered from y(t-1)=f1(X(t-1))+e ) \n')
fprintf('g=LSTM machine \n')
fprintf('f1=States to output (room temperature) machine (Pure weather conditions) \n')
fprintf('f2=Augmented states (with control inputs) to room temperature \n')

fprintf('set: \n')
fprintf('y(1)=y(t-1) \n')
fprintf('X(1)=X(t-1) \n' )
fprintf('u(1)=u(t-1) \n')
fprintf('Do t= 1: Horizon: \n')
fprintf('y(t+1)=g(y(t))+n # Predict the future output given present output \n')
fprintf('y(t)=f1(X(t))+e # Predict current output with current states \n')
fprintf('ybig(t,:)=y(t) \n')
fprintf('u(opt)=argmin||r(t)-f2(X(t);u(t),X(t))||+z # Optimise the control at time t \n')
fprintf('ubig(t,:)=u(opt) \n')
fprintf('Xbig(t,:)=X \n')

fprintf('X(opt)=argmin||y(t+1)-f1(X(t)||+z # Optimise the state at time t+1 \n')
fprintf('set X(t)= X(t+1)=X(opt) \n')
fprintf('set u(t)= u(t+1)=u(opt) \n')  
fprintf('End Do \n')

%%}
%%
disp('*******************************************************************')
oldfolder=cd;
cd(oldfolder);
addpath('Data');
addpath('Miscellaneous');
%% 
disp('-----------------------TRAIN STATES MACHINE---------------------')
folder = strcat('MLSL_machine_1');
mkdir(folder);
ds=tabularTextDatastore("Box.csv");
T=readall(ds);
T(:,1)=[];
Bt=T{:,:};
X=Bt(:,1:end-1);
yb=Bt(:,end);
epoch=3000;
batch_size=5;
[costtrain1,costtest1,rangees1,clfx1,clfy1,Model1]=Learning_Clement...
    (X,yb,epoch,batch_size,folder,oldfolder);

cd(folder)
save('rangees1.mat', 'rangees1');
save ('clfx1.mat', 'clfx1');
save ('clfy1.mat', 'clfy1');
save Model1.mat Model1
cd(oldfolder)

%%
disp('-----------------------TRAIN CONTROLLER MACHINE---------------------')
folder2 = strcat('MLSL_machine_2');
mkdir(folder2);
ds=tabularTextDatastore("GSHP.csv");
T=readall(ds);
T(:,1)=[];
Bt=T{:,:};
X=Bt(:,1:end-1);
yb=Bt(:,end);
[costtrain2,costtest2,rangees2,clfx2,clfy2,Model2]=Learning_Clement(X,...
    yb,epoch,batch_size,folder2,oldfolder);
cd(folder2)
save('rangees2.mat', 'rangees2');
save ('clfx2.mat', 'clfx2');
save ('clfy2.mat', 'clfy2');
save ('Model2.mat', 'Model2');
cd(oldfolder)
%%
disp('-----------------------TRAIN LSTM MACHINE---------------------')
folder3 = strcat('LSTM_machine');
mkdir(folder3);
cd('Data')
ds=tabularTextDatastore("Box.csv"); %Weather data from history till previous timestep
cd(oldfolder)
T=readall(ds);
T(:,1)=[];
output=T(:,end);
inpuut=T(:,1:end-1);
Bt=inpuut{:,:};
rngg=1;
shiftt=10;
[netLSTM,X_train,train_size,transs]=Learning_LSTM_Clement...
    (rngg,shiftt,Bt,oldfolder,folder3);
cd(folder3)
save('netLSTM.mat', 'netLSTM');
save('X_train.mat', 'X_train');
save('shiftt.mat', 'shiftt');
save('train_size.mat', 'train_size');
save('transs.mat', 'transs');
cd(oldfolder)
%%
rmpath('Data')
rmpath('Miscellaneous')
disp('-------------PROGRAMME EXECUTED----------------------------------')