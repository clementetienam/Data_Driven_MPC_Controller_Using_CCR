%%
clc;
clear;
close all;
disp('@author: Dr Clement Etienam')
%%
oldfolder=cd;
cd(oldfolder);
Resultss = 'Results';
mkdir(Resultss);
addpath('Miscellaneous');
addpath('LSTM_machine');


f1='MLSL_machine_1';
f2='MLSL_machine_2';
maxIter=input('Enter the Maximum iteration needed: ');
Nop=maxIter;
setpoint=20;
Nopin=1;
%for i =1:Nop
%% --------------PREDICT THE FUTURE STATES FROM THE LSTM------------------
disp('---------------------------Predict with the LSTM-------------------')
cd('LSTM_machine')
netLSTM=load('netLSTM.mat');
netLSTM=netLSTM.netLSTM;
X_train=load('X_train.mat');
X_train=X_train.X_train;
shiftt=load('shiftt.mat');
shiftt=shiftt.shiftt;
train_size=load('train_size.mat');
train_size=train_size.train_size;
clfx=load('clfx.mat');
clfxlstm=clfx.clfx;
cd(oldfolder)
%% Assume a sensor
cd('Data')
ds=tabularTextDatastore("Box.csv"); %Weather data from history till previous timestep
cd(oldfolder)
T=readall(ds);
T(:,1)=[];
output=T(:,end);
Bt1=output{:,:};    
True_temperature=Bt1(train_size+1:train_size+Nop,:); %True Room temperature
%True_temperature=Bt1;
%%
net2L=netLSTM;
net2B=netLSTM;
lastSteps=zeros(maxIter-1+shiftt,size(X_train,2));
aac=X_train(end-shiftt+1:end,:); %Last shiftt elements of weather 

cd(f1)
clfx=load('clfx1.mat');
clfx=clfx.clfx1;

clfy=load('clfy1.mat');
clfy=clfy.clfy1;

model=load('Model1.mat');
Model_4=model.Model1;

rangees=load('rangees1.mat');
rangees=rangees.rangees1;
cd(oldfolder)

% 
aac=clfxlstm.inverse_transform(aac);

sizec=size(rangees,2);
meanss=rangees(2,:);
meanss2=rangees(1,:);
N=input('Enter size of the ensemble: ');
hypps=Get_ensemble_2(N,size(rangees,2),meanss2,meanss,shiftt);
hyppsini=hypps;

for j=1:maxIter
fprintf(' %d|%d \n',j,maxIter);      
    if j==1
        ensemble=hyppsini;
    else
        ensemble=hyppsupdated;
    end
    Simm=[];
parfor i=1:N
    aa1=ensemble(:,i);
    aa1=reshape(aa1,[],9);
 Sim=Forwarding_LSTM2(aa1,...
    netLSTM,X_train,Nop,clfxlstm,shiftt,clfx,clfy,Model_4,1,j)   
 Simm(:,i)=reshape(Sim,[],1);   
end
[Ynew,temp(j,:)] = EnKF (ensemble,True_temperature(j,:), N, Simm);
hyppsupdated=Ynew;
end

tempmean=mean(temp,2);
Simbig=[];
parfor j=1:N
  fprintf(' %d|%d \n',j,N); 
     aa2=hyppsupdated(:,j);
     aa2=reshape(aa2,[],9); 
 Simmc=Forwarding_LSTM(aa2,...
    netLSTM,X_train,maxIter,clfxlstm,shiftt,clfx,clfy,Model_4,j )
  
  Simmc=reshape(Simmc,[],1);
%     Simmc=[];
%     aa=hyppsupdated(:,j);
%     aa=reshape(aa,[],9);
%     for k=1:200
%  Simc=Forwarding_LSTM2(aa,...
%     netLSTM,X_train,200,clfxlstm,shiftt,clfx,clfy,Model_4,1,k) ;  
%  Simmc(k,:)=reshape(Simc,[],1);   
%     end
    Simbig(:,j)=Simmc;
end

%% LSTM
     aac=reshape(aac,[],9); 
 Simmtrue=Forwarding_LSTM(aac,...
    netLSTM,X_train,maxIter,clfxlstm,shiftt,clfx,clfy,Model_4,j );

%% Mean
aamean=mean(hyppsupdated,2);
aamean=reshape(aamean,[],9); 
 Simmean=Forwarding_LSTM(aamean,...
    netLSTM,X_train,maxIter,clfxlstm,shiftt,clfx,clfy,Model_4,j );
%%

linecolor1=colordg(4);
figure()
 plot(Simbig(:,1:N),'Color',linecolor1,'LineWidth',2)
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 13);
 ylim([min(True_temperature) max(True_temperature)])
title('Performance','FontName','Helvetica', 'Fontsize', 13)
a = get(gca,'Children');
hold on
plot(True_temperature,'r','LineWidth',1)
b = get(gca,'Children');
hold on
plot(Simmtrue,'k','LineWidth',1)
b1 = get(gca,'Children');
hold on
plot(Simmean,'b','LineWidth',1)
b2 = get(gca,'Children');
hold on
plot(tempmean,'y','LineWidth',1)
b3 = get(gca,'Children');
 set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
h = [b3;b2;b1;b;a];
legend(h,'EnKF direct Mean','Mean','LSTM','True model','Realisations','location','northeast');