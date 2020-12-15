function tempoutt=Forwarding_LSTM2(aa,...
    netLSTM,X_train,steps,clfxlstm,shiftt,clfx,clfy,Model_4,i,j)
aa=(clfxlstm.transform(aa));
net2L=netLSTM;
net2B=netLSTM;
lastSteps=zeros(steps+shiftt,size(X_train,2));
lastSteps(1:shiftt,:) =aa; %The last shift elements of training data
lastSteps=lastSteps';
zz= X_train(1:end-shiftt,:); 
net2L = predictAndUpdateState(net2L, zz'); 
net2B = predictAndUpdateState(net2B, zz'); 

%% From time 0 to time needed

for i=1:steps
[net2L,lastSteps(:,i+shiftt)] =predictAndUpdateState(net2L,lastSteps(:,i));
end

pred_states = lastSteps(:,shiftt+1:end);
pred_states=clfxlstm.inverse_transform(pred_states');
% xForecast=(clfxlstm.transform(pred_states))';
% [net2B,forecastFromInput] =predictAndUpdateState(net2B,xForecast);
% pred_states=clfxlstm.inverse_transform(forecastFromInput');
X_test=(clfx.transform(pred_states));
zz = (predict(Model_4,X_test','ExecutionEnvironment','cpu'))';
tempout=clfy.inverse_transform(zz);
tempoutt=tempout(j,:);
end