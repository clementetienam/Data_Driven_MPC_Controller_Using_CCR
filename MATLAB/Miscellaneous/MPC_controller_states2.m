function [predicted_states,Simmini]=MPC_controller_states2(oldfolder,ff,...
    loop_clement,ensemble,Result_folder,y_train,pred_ini,N)
%%
cd(oldfolder);
pred_ini=double(pred_ini);
addpath(ff);

%%

clfx=load('clfx1.mat');
clfx=clfx.clfx1;

clfy=load('clfy1.mat');
clfy=clfy.clfy1;

model=load('Model1.mat');
Model_4=model.Model1;

rangees=load('rangees1.mat');
rangees=rangees.rangees1;
%%
%% Read Training data
sizec=size(rangees,2);
meanss=rangees(2,:);
meanss2=rangees(1,:);

cd(oldfolder)
sd=1;
rng(sd); % set random number generator
 %% Initial Guess;
Nop=size(y_train,1); % Number of points
%Nop=10;
szss=Nop;
disp('*******************************************************************')
if loop_clement > 2
error('Wrong choice please select 1-2')
end
%%
switch loop_clement
    case 1   
p=2;

% parfor jj=1:sizec
 % aj=meanss2(:,jj)+ (meanss(:,jj)- meanss2(:,jj))*sum(rand(szss,p),2)/p;
 % hyp_inipuree(:,jj) = reshape(aj,[],1);    
 %  end
 
 hyp_inipuree=double(ensemble);
technique=2; %1= fmincon 2=fminsearch;
%% Constrained optimization Algorithm

options = optimoptions('fmincon','Display',...
    'iter','MaxIter',...
10000,'TolX',10^-200,'TolFun',10^-200,'MaxFunEvals',...
10000,'PlotFcns',@optimplotfval,'UseParallel',true);

options2=optimset('Display','iter','MaxIter',...
10000,'TolX',10^-200,'TolFun',10^-200,'MaxFunEvals',...
10000,'PlotFcns',@optimplotfval,'UseParallel',true);  


parfor i=1:szss 
 hyp_inipure=hyp_inipuree(i,:);
%hyp_inipure=abs(hyp_inipure);


hypsgs=zeros(1,sizec);

hyp_ini=reshape(hyp_inipure,[],sizec);

switch technique
    case 1

FitnessFunction = @(x)Optimize_clement_LBFGS(x,Model_4,...
    y_train(i,:),clfx,clfy,sizec);
[hyp_updatedGA,fval] = fmincon(FitnessFunction...
    ,hyp_ini,[],[],[],[],meanss2,meanss,[],options);

    case 2
hyp_updatedGA=fminsearch('Optimize_clement_LBFGS',hyp_ini,...
   options2,Model_4,y_train(i,:),clfx,clfy,sizec);
end

hyp_unchangedGA=reshape(hyp_updatedGA,[],sizec);

for iy=1:sizec

av=hyp_updatedGA(:,iy);
jlow=meanss2(:,iy);
jup=meanss(:,iy);
av(av<=jlow)=jlow;
av(av>=jup)=jup;
hypsgs(:,iy)=av;
end

hyp_updatedGA=hypsgs;

X_proposed_lbfgsGA(i,:)=hyp_updatedGA;
X_proposed_lbfgs_unchangedGA(i,:)=hyp_unchangedGA;
cd(oldfolder);
 fprintf('Done %d | %d .\n', i,szss); 
end
%%
%% INITIAL
X_ini=(clfx.transform(hyp_inipuree(1:Nop,:)));

Simm_ini= reshape((predict(Model_4,X_ini'))',[],1);
Simm_ini=clfy.inverse_transform(Simm_ini);

%% Models

X_proposed_GA=(clfx.transform(X_proposed_lbfgsGA));

Simm_GA=reshape((predict(Model_4,X_proposed_GA'))',[],1);
Simm_GA=clfy.inverse_transform(Simm_GA);

X_proposed_lbfgs_unchangedaGA=(clfx.transform...
    (X_proposed_lbfgs_unchangedGA));
Simm_GA_unchanged=reshape((predict(Model_4,...
    X_proposed_lbfgs_unchangedaGA'))',[],1);
Simm_GA_unchanged=clfy.inverse_transform(Simm_GA_unchanged);
%% Compute Error with the True signal
for i=1:size(y_train,2)
%% Initial model
Error_ini(:,i)=immse(double(Simm_ini(:,i)),y_train(1:Nop,i));
Error_ini=sum(Error_ini./size(y_train(1:Nop,i),1));    
    
%%   Neil-Hadder       
Error_GA(:,i)=immse(double(Simm_GA(:,i)),y_train(1:Nop,i));
Error_GA=sum(Error_GA./size(y_train(1:Nop,i),1));

Error_GA_unchanged(:,i)=immse(double(Simm_GA_unchanged(:,i))...
    ,y_train(1:Nop,i));
Error_GA_unchanged=sum(Error_GA_unchanged....
    /size(y_train(1:Nop,i),1));
end
%%
%%
%%


Xini=(clfx.transform...
    (pred_ini));
Simmini=reshape((predict(Model_4,...
    Xini'))',[],1);
Simmini=clfy.inverse_transform(Simmini);






%predicted_states=X_proposed_lbfgsGA;
predicted_states=X_proposed_lbfgs_unchangedGA;




	case 2
%N=input('Enter size of the ensemble: ');
hypps=Get_ensemble_2(N,sizec,meanss2,meanss,Nop);
hyppsini=hypps;
 suni=sizec;
%% methodInfo
% localization setting
methodInfo.doLoc=0; % do localization? (1 = yes)
if methodInfo.doLoc
    % localization is not included here, but can be done if needed
    % see, for example, the recent paper
    % "Automatic and adaptive localization for ensemble-based history
    % matching", JPSE, 2019. https://doi.org/10.1016/j.petrol.2019.106559
    % and the references therein
end
% configuration of the iterative ensemble smoother, for more information, see
% "Iterative ensemble smoother as an approximate solution to a regularized 
% minimum-average-cost problem: theory and applications", SPE J. 2015.
% https://www.onepetro.org/journal-paper/SPE-176023-PA
methodInfo.beta = 0; % $beta$ determines the threshold value in one of the stopping criteria
methodInfo.maxOuterIter = 20; % maximum iteration number in the outer loop
methodInfo.maxInnerIter = 5; % maximum iteration number in the inner loop
methodInfo.init_lambda = 1; % initial lambda value
methodInfo.lambda_reduction_factor = 0.9; % reduction factor in case to reduce gamma
methodInfo.lambda_increment_factor = 2; % increment factor in case to increase gamma 
methodInfo.doTSVD = 1; % do a TSVD on the cov of simulated obs? (1 = yes)
obv_nLevel = 1;
if methodInfo.doTSVD
    methodInfo.tsvdCut = 0.99; % discard eigenvalues/eigenvectors if they are not among the truncated leading ones
end
methodInfo.min_RN_change = 1; % minimum residual norm (RN) change (in percentage); RN(k) - RN(k+1) > RN(k) * min_RN_change / 100
 ensemble = iES(methodInfo,...
    y_train,hyppsini,clfx,clfy,sizec,oldfolder,Model_4);
	
use_ensemble=ensemble(:,1:end-1);

 %% Rectify for the Limits of the Machine
 parfor ii=1:N
 aj=use_ensemble(:,ii);
  aa=aj;
aa=reshape(aa,[],suni);
jxv3=zeros(size(aa));
 for i=1:sizec
av1=aa(:,i);
jlow=meanss2(:,i);
jup=meanss(:,i);
av1(av1<=jlow)=jlow;
av1(av1>=jup)=jup;
jxv3(:,i)=av1;
 end
 hyp_might(:,:,ii)=jxv3;
 end
 
lastt=ensemble(:,end);
lastt=reshape(lastt,[],suni);
hyp_mean=(mean(hyp_might,3));
hyp_mean_unchanged=lastt;

predicted_states=lastt;
aa=hyp_mean;
spit=abs((Forwarding(aa,clfx,...
clfy,oldfolder,Model_4)));
spitmean=reshape(spit,[],size(y_train,2));


aa=hyp_mean_unchanged;
spit=abs((Forwarding(aa,clfx,...
clfy,oldfolder,Model_4)));
spitmeanraw=reshape(spit,[],size(y_train,2));



Xini=(clfx.transform...
    (pred_ini));
Simmini=reshape((predict(Model_4,...
    Xini'))',[],1);
Simmini=clfy.inverse_transform(Simmini);

end
end
