function [netLSTM,XTrain,train_size,transs]=Learning_LSTM_Clement(rngg,shiftt,Bt,oldfolder,folder)
%% Train LSTM Network 
rng(rngg)
data=Bt;
numTimeStepsTrain = floor(0.5*length(data));
train_size=numTimeStepsTrain; 
dataTrain = data(1:numTimeStepsTrain+1,:);
dataTest = data(numTimeStepsTrain+1:end,:);
mu = mean(dataTrain,1);
sig = std(dataTrain,1);
dataTrainStandardized = (dataTrain - mu) ./ sig;
shift=shiftt;
XTrain = dataTrainStandardized(1:end-shift,:);
YTrain = dataTrainStandardized(shift+1:end,:);

transs=[mu;sig];
numFeatures = size(XTrain,2);
numResponses = size(YTrain,2);
numHiddenUnits = 300;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',1050, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize', 1 , ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1, ...
    'Plots','training-progress');
netLSTM = trainNetwork(XTrain',YTrain',layers,options);
net2=netLSTM;


dataTestStandardized = (dataTest - mu) ./ sig;
XTest = dataTestStandardized(1:end-shift,:);
net2 = predictAndUpdateState(net2,XTrain(1:end-shift,:)');
%% 
%%
lastSteps = zeros(size(XTest,1)+shift,size(data,2)) ;
lastSteps(1:shift,:) = XTrain(end-shift+1:end,:); %The last shift elements of training data
lastSteps=lastSteps';
for i=1:size(XTest,1)
[net2,lastSteps(:,i+shift)] =predictAndUpdateState(net2,lastSteps(:,i));
end

forecastFromSelf = lastSteps(:,1:size(XTest,1));
forecastFromSelf=forecastFromSelf';
forecastFromSelf = sig.*forecastFromSelf+ mu;


XTest2=sig.*XTest+ mu;
XTrain2=sig.*XTrain+ mu;

%%
figure(1)
for i=1:size(data,2)
subplot(3,3,i)
plot(XTrain2(:,i),'r','LineWidth',1)
hold on
plot(XTest2(:,i) ,'b','LineWidth',1)
hold on
plot(forecastFromSelf(:,i) ,'g','LineWidth',1)
hold on
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title( strcat('X', sprintf('%d',i)),...
    'FontName','Helvetica', 'Fontsize', 9)
legend('True train','True test','ForecastSelf test 1',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end

cd(folder)
%Namefilef=strcat('performance_', sprintf('%d',jjm),'.fig');
saveas(gcf,'Performace.fig')
cd(oldfolder)
end