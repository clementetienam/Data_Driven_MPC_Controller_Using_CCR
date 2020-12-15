function pred_states=States_prediction_clement(shift,steps,X_train,net,lastshiff)
net2=net;
totalLength=shift+steps;
lastSteps = zeros(totalLength,size(X_train,2)) ;
% In realistic scenarios, get the last shift temperature from weather data

lastSteps(1:shift,:) = lastshiff; %The last shift elements of training data
lastSteps=lastSteps';
% Here just assign the last shifft elemts here
zz= X_train(1:end-shift,:); %All elements of training data apart from last shift elements
%This will simply be the whole training data
net2 = predictAndUpdateState(net2, zz'); 

% This is the self prediction loop
for i=1:steps
[net2,lastSteps(:,i+shift)] =predictAndUpdateState(net2,lastSteps(:,i));
end
pred_states = lastSteps(:,1:steps);
%pred_states=lastSteps(:,shift+1:end);
pred_states=pred_states';
end