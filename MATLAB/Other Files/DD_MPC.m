%%
clc;
clear;
close all;
disp('@author: Dr Clement Etienam')
%%
oldfolder=cd;
cd(oldfolder);
Resultss = 'Results_Batch';
mkdir(Resultss);
addpath('Miscellaneous');
addpath('LSTM_machine');
disp('SELECT OPTION FOR OPTMSATION')
disp('1:LBFGS')
disp('2:I-ES')
clement=input('Enter the optimisation scheme desired: ');
if (clement > 2) || (clement < 1)
error('Wrong choice please select 1-2')
end
f1='MLSL_machine_1';
f2='MLSL_machine_2';
Nop=input('Enter time step for forward prediction required (20:40): ');

%% Choose Data Type
Datatype=2;
if Datatype==1
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
%%
methodd=input('Enter 1= open loop; 2=closed loop: ');
if (methodd > 2) || (methodd < 1)
error('Wrong choice please select 1-2')
end
%tobesure=1; %0
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
% In realistic scenarios we should pass on the last shifft elements
% recorded also, In that case X_train will be whole
disp('1:No')
disp('2:Yes')
noo=input('Do you have historical data?: ');
if (noo > 2) || (noo < 1)
error('Wrong choice please select 1-2')
end
if noo==1 %No information to prior weather states
    aa=X_train(end-shiftt+1:end,:); %Last shiftt elements of weather 
    %aa=(clfx.transform(aa));
else 
%Information on prior shiftt temperature
% Insert Last shifft days here from historical data if available
cd('Data')
ds=tabularTextDatastore("Box.csv"); %Weather data from history till previous timestep
cd(oldfolder)
T=readall(ds);
T(:,1)=[];
output=T(:,end);
inpuut=T(:,1:end-1);
Bt=inpuut{:,:};
Bt1=output{:,:};    
%n = 1;
%R = [200 length(Bt,1)-shiftt];
%z = floor(rand(n,1)*range(R)+min(R));
aa=Bt(train_size+1:train_size+shiftt,:);
aa= (Bt - mu) ./ sig;
end
%Truetemp=Bt(z+shiftt+1: z+shiftt+Nop,:);
pred_states=States_prediction_clement(shiftt,Nop,X_train,netLSTM,aa);
pred_states=transs(2,:).*pred_states+ transs(1,:);
%
%if tobesure==1
%net2=netLSTM;
%zz= X_train(1:end-shiftt,:); %All elements of training data apart from last shift elements
%net2 = predictAndUpdateState(net2, zz'); 
%xForecast=(clfx.transform(pred_states))';
%for i =1:size(xForecast,2)
%     [net2,forecastFromInput(:,i)] =predictAndUpdateState(net2,xForecast(:,i));
 %end
%pred_states=clfx.inverse_transform(forecastFromInput');
%end
%
pred_ini=pred_states;

if methodd==1
 disp('Open loop and no feedback')
optimised_states=pred_states;
cd(f1)
clfx=load('clfx1.mat');
clfx=clfx.clfx1;

clfy=load('clfy1.mat');
clfy=clfy.clfy1;

model=load('Model1.mat');
Model_4=model.Model1;
cd(oldfolder)
parameters=[optimised_states];
X_test=(clfx.transform(parameters));
zz = (predict(Model_4,X_test','ExecutionEnvironment','cpu'))';
zzstatestemp=clfy.inverse_transform(zz);

cd('Data')
ds=tabularTextDatastore("Box.csv"); %Weather data from history till previous timestep
cd(oldfolder)
T=readall(ds);
T(:,1)=[];
output=T(:,end);
Bt1=output{:,:};    
True_temperature=Bt1(train_size+1:train_size+Nop,:); %True Room temperature
figure()
plot(True_temperature,'r','LineWidth',1)
hold on
plot(zzstatestemp,'k--','LineWidth',1)
hold on
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 13);
title('Temperature Trend','FontName','Helvetica', 'Fontsize', 13)
legend('True room Temperature','Predicted temperature from LSTM',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')
cd(Resultss)
if clement==1
saveas(gcf,'Predictiontempopen_LBFGS','fig')
else
saveas(gcf,'Predictiontempopen_I-ES','fig')
end
cd(oldfolder)
else
disp('closed loop: we have information to the fedback true room temperature')
cd('Data')
ds=tabularTextDatastore("Box.csv"); %Weather data from history till previous timestep
cd(oldfolder)
T=readall(ds);
T(:,1)=[];
output=T(:,end);
Bt1=output{:,:};    
True_temperature=Bt1(train_size+1:train_size+Nop,:); %True Room temperature
optimised_states=MPC_controller_states(oldfolder,...
    f1,clement,pred_states,Resultss,True_temperature,pred_ini);
end
%% ---------------OPTIMISE FOR THE SET POINT TEMPERATURE------------------
cd('Data')
ds=tabularTextDatastore("GSHP.csv"); %Weather data from history till previous timestep
cd(oldfolder)
T=readall(ds);
T(:,1)=[];
output=T(:,end);
Bt=output{:,:};    
n = 1;
%yy=Bt(train_size+1:train_size+Nop,:); %% Set point temperature
%for i=1:Nop
 %   yy(i,:)=20;
%end

[optimised_states_control,summary]=MPC_Controller_states_controll(oldfolder,...
    f2,clement,yy,Resultss,optimised_states);
%% ---------------------SAVE FILES AFTER OPTIMISATION----------------------
optimisedd=[optimised_states,optimised_states_control];
%optimisedd=[optimised_states_control];
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
Namefile2=strcat('states_controller_LBFGS_batch','.csv');
else
Namefile2=strcat('states_controller_IES_batch','.csv');
end
% headers2={'GSHPCLG:Heat Pump Electric Power [W](Daily)',...
%     'GSHPCLG:Heat Pump Source Side Inlet Temperature [C](Daily)',...
%     'GSHPHEATING:Heat Pump Electric Power [W](Daily)',...
%     'GSHPHEATING:Heat Pump Source Side Inlet Temperature [C](Daily) '}; 
% headers3 = {'Environment:Site Outdoor Air Drybulb Temperature [C](Daily)',...
%     'Environment:Site Outdoor Air Wetbulb Temperature [C](Daily)',...
%     'Environment:Site Outdoor Air Relative Humidity [%](Daily)',...
%     'Environment:Site Wind Speed [m/s](Daily)',...
%     'Environment:Site Wind Direction [deg](Daily)',...
%     'Environment:Site Horizontal Infrared Radiation Rate per Area [W/m2](Daily)',...
%     'Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](Daily)',...
%     'Environment:Site Direct Solar Radiation Rate per Area [W/m2](Daily)',...
%     'THERMAL ZONE: BOX:Zone Outdoor Air Wind Speed [m/s](Daily)'}; 

cd(Resultss)
csvwrite_with_headers(  Namefile2,Matrix,headers);
% csvwrite_with_headers(  'controller.csv',optimised_states_control,headers2);
%csvwrite_with_headers( 'Predicted_states.csv',optimised_states,headers3);
%csvwrite_with_headers( 'optimised_states.csv',...
  %  optimised_states_control(:,1:9),headers3);
cd(oldfolder)
rmpath('Miscellaneous')
rmpath('MLSL_machine_2');
rmpath('MLSL_machine_1');
rmpath('LSTM_machine')