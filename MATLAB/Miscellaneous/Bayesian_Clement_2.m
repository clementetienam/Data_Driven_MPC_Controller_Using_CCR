function DupdateK=Bayesian_Clement_2(hypps,N,clfx,clfy,...
    oldfolder,ytrue,alpha,suni,model)
parfor i=1:N
    aa=(hypps(:,i));
	aa=reshape(aa,[],suni)
	spit=abs((Forwarding(aa,clfx,clfy,oldfolder,model)));
	spit=reshape(spit,[],1);
Sim(:,i)=spit;
end
[DupdateK] = ESMDA (hypps,reshape(ytrue,[],1), N, Sim,alpha);
end