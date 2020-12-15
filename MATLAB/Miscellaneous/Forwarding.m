function zz=Forwarding(parameters,clfx,...
    clfy,oldfolder,model)
X_test=(clfx.transform(parameters));
cd(oldfolder);
zz = (predict(model,X_test','ExecutionEnvironment','cpu'))';
zz=clfy.inverse_transform(zz);
%zz=reshape(zz,[],1);

end