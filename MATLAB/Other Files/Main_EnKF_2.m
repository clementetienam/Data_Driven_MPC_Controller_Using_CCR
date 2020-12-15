%%
clc;
clear;
close all;
disp('@author: Dr Clement Etienam')
%%
oldfolder=cd;
cd(oldfolder);
addpath('Miscellaneous');
addpath('LSTM_machine');


f1='MLSL_machine_1';
f3='MLSL_machine_3';
% maxIter=input('Enter the Maximum iteration needed: ');
% Nop=maxIter;
% setpoint=20;
% Nopin=1;
%for i =1:Nop
%% Assume a sensor
% cd('Data')
% ds=tabularTextDatastore("Box.csv"); %Weather data from history till previous timestep
% cd(oldfolder)
% T=readall(ds);
% T(:,1)=[];
% output=T(:,end);
% Bt1=output{:,:};    
% True_temperature=Bt1(train_size+1:train_size+Nop,:); %True Room temperature
%True_temperature=Bt1;
%%


cd(f3)
clfx=load('clfx3.mat');
clfx=clfx.clfx3;

clfy=load('clfy3.mat');
clfy=clfy.clfy3;

model=load('Model3.mat');
Model_4=model.Model3;

rangees=load('rangees3.mat');
rangees=rangees.rangees3;
cd(oldfolder)

cd('Data')
ds=tabularTextDatastore("Box.csv");
T=readall(ds);
T(:,1)=[];
Bt=T{:,:};
cd(oldfolder)
input=Bt(1,:);

%%
aac=clfx.transform(input);
aac1=aac;
i=1;
for i=1:size(Bt,1)-1
  aac = (predict(Model_4,aac','ExecutionEnvironment','cpu'))';  
  zz(i,:)=aac;
end
zz=[aac1;zz];
zz=clfy.inverse_transform(zz);

% % 
% aac=clfxlstm.inverse_transform(aac);
% 
% sizec=size(rangees,2);
% meanss=rangees(2,:);
% meanss2=rangees(1,:);
% N=input('Enter size of the ensemble: ');
% hypps=Get_ensemble_2(N,size(rangees,2),meanss2,meanss,shiftt);
% hyppsini=hypps;
% 
% for j=1:maxIter
% fprintf(' %d|%d \n',j,maxIter);      
%     if j==1
%         ensemble=hyppsini;
%     else
%         ensemble=hyppsupdated;
%     end
%     Simm=[];
% parfor i=1:N
%     aa1=ensemble(:,i);
%     aa1=reshape(aa1,[],9);
%  Sim=Forwarding_LSTM2(aa1,...
%     netLSTM,X_train,Nop,clfxlstm,shiftt,clfx,clfy,Model_4,1,j)   
%  Simm(:,i)=reshape(Sim,[],1);   
% end
% [Ynew] = EnKF (ensemble,True_temperature(j,:), N, Simm);
% hyppsupdated=Ynew;
% end