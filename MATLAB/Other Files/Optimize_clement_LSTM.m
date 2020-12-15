function out=Optimize_clement_LSTM(parameters,...
    netLSTM,X_train,steps,...
   clfxlstm,shiftt,clfx,clfy,Model_4,ytrue,i)


aa=reshape(parameters,[],9);
zz=Forwarding_LSTM(aa,...
    netLSTM,X_train,steps,clfxlstm,shiftt,clfx,clfy,Model_4,i);

%%

ytrue=reshape(ytrue,[],1);
Hardmean=double(reshape(zz,[],1));
gg=size(ytrue,1);
% a1=((ytrue'-Hardmean').^2).^0.5;
a1=(1/(2*gg)) * sum((ytrue-Hardmean).^2);

%a1=abs((ytrue-Hardmean));

cc = sum(a1);
out=cc;

end