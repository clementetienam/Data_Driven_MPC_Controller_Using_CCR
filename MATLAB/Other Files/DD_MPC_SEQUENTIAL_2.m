%%
clc;
clear;
close all;
disp('@author: Dr Clement Etienam')
%%
oldfolder=cd;
cd(oldfolder);
Resultss = 'Results_Sequential';
mkdir(Resultss);
addpath('Miscellaneous');
addpath('LSTM_machine');
disp('SELECT OPTION FOR OPTMSATION')
disp('1:LBFGS')
disp('2:I-ES')
%disp('3 :ES-MDA')
clement=input('Enter the optimisation scheme desired: ');
if clement==1
   %N=input('Enter size of the ensemble: '); 
N=200;
else
   %N=input('Enter size of the ensemble: '); 
N=200;
end

f1='MLSL_machine_1';
f2='MLSL_machine_2';
maxIter=input('Enter the Maximum iteration needed (20-40): ');
%maxIter=20;
Nop=maxIter;
%% Choose Data Type
DataType=1;
if DataType==1
x=linspace(1,Nop,Nop)';
x=x./Nop;
y=zeros(size(x,1),1);
for i=1:Nop
if (x(i,:)>=1/Nop) && (x(i,:)<2/Nop)
y(i,:)=17;
elseif (x(i,:)>5/Nop) && (x(i,:)<Nop/Nop)
y(i,:)=22;
else
y(i,:)=20;
end
end
setpoint=reshape(y,[],1);
yy=setpoint;
else
Nop =35;
X = linspace(0,4*pi,Nop);
Y = sin(X);
[xb,yb] = stairs(Y);
a=min(yb);
b=max(yb);

yy=((4.*yb)+21);
yy=yy(1:Nop,:);
setpoint=yy;
end
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
transs=load('transs.mat');
transs=transs.transs;
cd(oldfolder)
%% Assume a sensor
cd('Data')
ds=tabularTextDatastore("Box.csv"); %Weather data from history till previous timestep
cd(oldfolder)
T=readall(ds);
T(:,1)=[];
output=T(:,end);
Bt1=output{:,:};    
True_temperature=Bt1(train_size+1:train_size+Nop,:); %True Room temperature % This should be from a serperate model like co-simulation
%%
net2L=netLSTM;
net2B=netLSTM;
lastSteps=zeros(maxIter-1+shiftt,size(X_train,2));
aa=X_train(end-shiftt+1:end,:); %Last shiftt elements of weather 
lastSteps(1:shiftt,:) = aa; %The last shift elements of training data
lastSteps=lastSteps';
zz= X_train(1:end-shiftt,:); 
net2L = predictAndUpdateState(net2L, zz'); 
net2B = predictAndUpdateState(net2B, zz'); 


%% From time 0 to time 1
[net2L,lastStepss(:,shiftt+1)] =predictAndUpdateState(net2L,lastSteps(:,end));
%pred_states = lastSteps(:,shiftt+1);
pred_states=lastStepss(:,end);
pred_states=(pred_states');
pred_states=transs(2,:).*pred_states+ transs(1,:);
pred_ini=pred_states;
optimised_states=pred_states;	


%%
i=1;
while i <= maxIter
fprintf(' %d|%d \n',i,maxIter);  

spit_states(i,:)=optimised_states;

%% ---------------OPTIMISE FOR THE SET POINT TEMPERATURE------------------
[optimised_states_control(i,:),summary(i,:),Temp_control(i,:)]...
    =MPC_Controller_states_controll2...
    (oldfolder,...
    f2,clement,setpoint(i,:),Resultss,optimised_states,N);

%Siying=[optimised_states,optimised_states_control(i,:)];
% Pass variable Siying to siying to get Temperature
%% Sensor room temperature arrives
%Get the true temperature from Siyin
[optimised_states,tempp]=MPC_controller_states2(oldfolder,...
    f1,clement,pred_states,Resultss,True_temperature(i,:),pred_ini,N);

pred_temp(i,:)=tempp;
spit_states(i,:)=optimised_states;
pred_ini=optimised_states;
pred_states=optimised_states;
i=i+1;
end

%% Plot outside to see how good the sequential controller was
cd(f2);
clfx=load('clfx2.mat');
clfx=clfx.clfx2;

clfy=load('clfy2.mat');
clfy=clfy.clfy2;
model=load('Model2.mat');
Model_4=model.Model2;
cd(oldfolder);
parameters=[spit_states,optimised_states_control];
X_test=(clfx.transform(parameters));
zz = (predict(Model_4,X_test','ExecutionEnvironment','cpu'))';
zzcontrol=clfy.inverse_transform(zz);
%% Plot How good to see how good the temperature prediction was
cd(f1)
clfx=load('clfx1.mat');
clfx=clfx.clfx1;

clfy=load('clfy1.mat');
clfy=clfy.clfy1;

model=load('Model1.mat');
Model_4=model.Model1;
cd(oldfolder)
parameters=[spit_states];
X_test=(clfx.transform(parameters));
zz = (predict(Model_4,X_test','ExecutionEnvironment','cpu'))';
zzstatestemp=clfy.inverse_transform(zz);

figure()
subplot(2,2,1)
plot(True_temperature,'r','LineWidth',1)
hold on
plot(zzstatestemp,'k--','LineWidth',1)
hold off
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 13);
title('Temperature Trend','FontName','Helvetica', 'Fontsize', 13)
legend('True room Temperature','optimised temperature','location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')

setpointt=setpoint;

subplot(2,2,2)
plot(setpointt,'r','LineWidth',1)
hold on
plot(zzcontrol,'k--','LineWidth',1)
hold off
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 13);
title('Temperature Trend','FontName','Helvetica', 'Fontsize', 13)
legend(' Set point temperature ','Controller','location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')

cd(Resultss)
if clement==1
saveas(gcf,'optimisedLBFGS','fig')
else
saveas(gcf,'optimisedI-ES','fig')
end
cd(oldfolder)

%%
%%
% %% ---------------------SAVE FILES AFTER OPTIMISATION----------------------
optimisedd=[spit_states,optimised_states_control];
Matrix=optimisedd;
headers = {'Environment:Site Outdoor Air Drybulb Temperature [C](Daily)',...
    'Environment:Site Outdoor Air Wetbulb Temperature [C](Daily)',...
    'Environment:Site Outdoor Air Relative Humidity [%](Daily)',...
    'Environment:Site Wind Speed [m/s](Daily)',...
    'Environment:Site Wind Direction [deg](Daily)',...
    'Environment:Site Horizontal Infrared Radiation Rate per Area [W/m2](Daily)',...
    'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Daily)',...
    'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Daily)',...
    'THERMAL ZONE: BOX:Zone Outdoor Air Wind Speed [m/s](Daily)',...
    'GSHPCLG:Heat Pump Electric Power [W](Daily)',...
    'GSHPCLG:Heat Pump Source Side Inlet Temperature [C](Daily)',...
    'GSHPHEATING:Heat Pump Electric Power [W](Daily)',...
    'GSHPHEATING:Heat Pump Source Side Inlet Temperature [C](Daily) '};
if clement==1
Namefile2=strcat('states_controller_LBFGS','.csv');
else
Namefile2=strcat('states_controller_IES','.csv');
end
cd(Resultss)
csvwrite_with_headers(  Namefile2,Matrix,headers);
cd(oldfolder)
rmpath('Miscellaneous')
rmpath('MLSL_machine_2');
rmpath('MLSL_machine_1');
rmpath('LSTM_machine')