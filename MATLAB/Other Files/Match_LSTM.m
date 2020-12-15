%%
clc;
clear;
close all;
disp('@author: Dr Clement Etienam')
%%
oldfolder=cd;
cd(oldfolder);
% Resultss = 'Results';
% mkdir(Resultss);
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
%%
net2L=netLSTM;
net2B=netLSTM;
lastSteps=zeros(maxIter-1+shiftt,size(X_train,2));
aa=X_train(end-shiftt+1:end,:); %Last shiftt elements of weather 




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
aa=clfxlstm.inverse_transform(aa);

methodd=1;

if methodd==1
options2=optimset('Display','iter','MaxIter',...
10000,'TolX',10^-200,'TolFun',10^-200,'MaxFunEvals',...
10000,'PlotFcns',@optimplotfval,'UseParallel',true);  
tempini=Forwarding_LSTM(aa,...
    netLSTM,X_train,Nop,clfxlstm,shiftt,clfx,clfy,Model_4,1);



% for i=1:Nop
% hyp_updated=fminsearch('Optimize_clement_LSTM',aa,...
%    options2,netLSTM,X_train,i,...
%    clfxlstm,shiftt,clfx,clfy,Model_4,True_temperature(i,:),i);
% end


hyp_updated=fminsearch('Optimize_clement_LSTM',aa,...
   options2,netLSTM,X_train,Nop,...
   clfxlstm,shiftt,clfx,clfy,Model_4,True_temperature,1);

%%
%for i=1:Nop
tempfinal=Forwarding_LSTM(hyp_updated,...
    netLSTM,X_train,Nop,clfxlstm,shiftt,clfx,clfy,Model_4,1);
%end

else
sizec=size(rangees,2);
meanss=rangees(2,:);
meanss2=rangees(1,:);

N=input('Enter size of the ensemble: ');
hypps=Get_ensemble_2(N,sizec,meanss2,meanss,shiftt);
hyppsini=hypps;
 suni=sizec;
%% methodInfo
% localization setting
methodInfo.doLoc=0; % do localization? (1 = yes)
if methodInfo.doLoc
    % localization is not included here, but can be done if needed
    % see, for example, the recent paper
    % "Automatic and adaptive localization for ensemble-based history
    % matching", JPSE, 2019. https://doi.org/10.1016/j.petrol.2019.106559
    % and the references therein
end
% configuration of the iterative ensemble smoother, for more information, see
% "Iterative ensemble smoother as an approximate solution to a regularized 
% minimum-average-cost problem: theory and applications", SPE J. 2015.
% https://www.onepetro.org/journal-paper/SPE-176023-PA
methodInfo.beta = 0; % $beta$ determines the threshold value in one of the stopping criteria
methodInfo.maxOuterIter = 10; % maximum iteration number in the outer loop
methodInfo.maxInnerIter = 3; % maximum iteration number in the inner loop
methodInfo.init_lambda = 1; % initial lambda value
methodInfo.lambda_reduction_factor = 0.9; % reduction factor in case to reduce gamma
methodInfo.lambda_increment_factor = 2; % increment factor in case to increase gamma 
methodInfo.doTSVD = 1; % do a TSVD on the cov of simulated obs? (1 = yes)
obv_nLevel = 1;
if methodInfo.doTSVD
    methodInfo.tsvdCut = 0.99; % discard eigenvalues/eigenvectors if they are not among the truncated leading ones
end
methodInfo.min_RN_change = 1; % minimum residual norm (RN) change (in percentage); RN(k) - RN(k+1) > RN(k) * min_RN_change / 100
ensemble = iESLSTM(methodInfo,...
    True_temperature,hypps,clfx,clfy,suni,Model_4,...
    netLSTM,X_train,Nop,...
   clfxlstm,shiftt);

lastt=ensemble(:,end);
lastt=reshape(lastt,[],9);
tempfinal=Forwarding_LSTM(lastt,...
    netLSTM,X_train,Nop,clfxlstm,shiftt,clfx,clfy,Model_4,1);

tempini=Forwarding_LSTM(aa,...
    netLSTM,X_train,Nop,clfxlstm,shiftt,clfx,clfy,Model_4,1);
end
figure()
plot(True_temperature,'r','LineWidth',1)
hold on
plot(tempfinal,'k--','LineWidth',1)
hold on
plot(tempini,'b--','LineWidth',1)
hold off
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 13);
title('Temperature Trend','FontName','Helvetica', 'Fontsize', 13)
legend(' True','optimised','initial','location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')