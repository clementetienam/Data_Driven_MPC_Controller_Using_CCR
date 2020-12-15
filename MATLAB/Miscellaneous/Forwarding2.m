function zz=Forwarding2(parameters,clfx,...
    clfy,oldfolder,model,augment)
parameters=[augment,parameters];
X_test=(clfx.transform(parameters));
cd(oldfolder);
zz = (predict(model,X_test','ExecutionEnvironment','cpu'))';
zz=clfy.inverse_transform(zz);
%zz=reshape(zz,[],1);

end