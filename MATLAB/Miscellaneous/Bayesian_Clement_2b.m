function DupdateK=Bayesian_Clement_2b(hypps,N,clfx,clfy,...
    oldfolder,ytrue,alpha,suni,model,augment)
parfor i=1:N
    aa=(hypps(:,i));
	aa=reshape(aa,[],suni)
	spit=abs((Forwarding2(aa,clfx,clfy,oldfolder,model,augment)));
	spit=reshape(spit,[],1);
Sim(:,i)=spit;
end
[DupdateK] = ESMDA (hypps,reshape(ytrue,[],1), N, Sim,alpha);
end