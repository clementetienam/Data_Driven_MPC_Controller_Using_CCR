function [costtrain,costtest,rangees,clfx,clfy,Model]= Learning_Clement(X,y,...
    epoch,batch_size,folder,oldfolder)

for ia=1:size(X,2)
maxclem(:,ia)=max(X(:,ia));
end

for ia=1:size(X,2)
minclem(:,ia)=min(X(:,ia));
end

rangees=[minclem;maxclem];


Xuse=X;
yuse=y;

clfx = MinMaxScaler();
(clfx.fit(Xuse));
Xuse=(clfx.transform(Xuse));



clfy = MinMaxScalery();

(clfy.fit(yuse));
yuse=(clfy.transform(yuse));


Test_percentage=0.2;
disp('')
[X_train, X_test, y_train, y_test,ind_train,ind_test] = train_test_split...
    (Xuse,yuse,Test_percentage);

input_count = size( Xuse , 2 );
output_count = size( yuse , 2 );

layers = [ ...
    sequenceInputLayer(input_count)
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(80)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(output_count)
    regressionLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',epoch, ...
    'MiniBatchSize', batch_size , ...
    'ValidationFrequency',10, ...
    'ValidationPatience',5, ...
    'Verbose',true, ...
    'Plots','training-progress');

Model = trainNetwork(X_train',y_train',layers,options);

[costtrain,costtest,T,y]=Check_Accuracy(X_train,X_test,y_train,...
    y_test,clfy,Model,oldfolder,folder,'performance_Supervised_Machine2.fig',...
    ind_train,ind_test);

end