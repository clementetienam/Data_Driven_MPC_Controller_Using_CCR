function [predicted_states,objss]=MPC_Controller_states_controll(oldfolder,ff,...
    loop_clement,y_train,Result_folder,augment)
%%
cd(oldfolder);
augment=double(augment);
addpath(ff);
%%

clfx=load('clfx2.mat');
clfx=clfx.clfx2;

clfy=load('clfy2.mat');
clfy=clfy.clfy2;

model=load('Model2.mat');
Model_4=model.Model2;

rangees=load('rangees2.mat');
rangees=rangees.rangees2;
%%
%% Read Training data

meanss=rangees(2,10:13);
meanss2=rangees(1,10:13);
sizec=size(meanss,2);
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
disp('Using LBFGS for Optimisation')        
p=2;

 parfor jj=1:sizec
  aj=meanss2(:,jj)+ (meanss(:,jj)- meanss2(:,jj))*sum(rand(szss,p),2)/p;
  hyp_inipuree(:,jj) = reshape(aj,[],1);    
   end
technique=2; %1= fmincon 2=fminsearch;
%% Constrained optimization Algorithm

options = optimoptions('fmincon','Display',...
    'iter','MaxIter',...
1000,'TolX',10^-200,'TolFun',10^-200,'MaxFunEvals',...
500,'PlotFcns',@optimplotfval,'UseParallel',true);

options2=optimset('Display','iter','MaxIter',...
1000,'TolX',10^-200,'TolFun',10^-200,'MaxFunEvals',...
500,'PlotFcns',@optimplotfval,'UseParallel',true);  


parfor i=1:szss 
 hyp_inipure=hyp_inipuree(i,:);
hyp_inipure=abs(hyp_inipure);


hypsgs=zeros(1,sizec);

hyp_ini=reshape(hyp_inipure,[],sizec);

switch technique
    case 1

FitnessFunction = @(x)Optimize_clement_LBFGS_2(x,Model_4,...
    y_train(i,:),clfx,clfy,augment(i,:),sizec);
[hyp_updatedGA,fval] = fmincon(FitnessFunction...
    ,hyp_ini,[],[],[],[],meanss2,meanss,[],options);

    case 2
hyp_updatedGA=fminsearch('Optimize_clement_LBFGS_2',hyp_ini,...
   options2,Model_4,y_train(i,:),clfx,clfy,augment(i,:),sizec);
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
aaini=[augment,hyp_inipuree(1:Nop,:)];
%aaini=hyp_inipuree(1:Nop,:);
X_ini=(clfx.transform(aaini));

Simm_ini= reshape((predict(Model_4,X_ini'))',[],1);
Simm_ini=clfy.inverse_transform(Simm_ini);

%% Models
aa2=[augment,X_proposed_lbfgsGA];
%aa2=X_proposed_lbfgsGA;
X_proposed_GA=(clfx.transform(aa2));

Simm_GA=reshape((predict(Model_4,X_proposed_GA'))',[],1);
Simm_GA=clfy.inverse_transform(Simm_GA);

aa3=[augment,X_proposed_lbfgs_unchangedGA];
%aa3=X_proposed_lbfgs_unchangedGA;
X_proposed_lbfgs_unchangedaGA=(clfx.transform...
    (aa3));
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

figure()
%%
%subplot(2,2,1);
plot(y_train(1:szss,1),'r','LineWidth',1)
hold on

plot(Simm_GA_unchanged(:,1),'b--','LineWidth',1)

hold off
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Temperature load','FontName','Helvetica', 'Fontsize', 13);
title('Temperature Trend','FontName','Helvetica', 'Fontsize', 13)
legend('Set point temperature','Batch Controller:LBFGS','location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')



% subplot(2,2,2);
% plot(y_train(1:szss,1),'r','LineWidth',1)
% hold on
% 
% plot(Simm_GA(:,1),'b--','LineWidth',1)
% hold off
% xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Heat load','FontName','Helvetica', 'Fontsize', 13);
% title('Heat Load Trend(constrained)','FontName','Helvetica', 'Fontsize', 13)
% legend('True model','Recovered Model','location','northeast');
% set(gca, 'FontName','Helvetica', 'Fontsize', 9)
% set(gcf,'color','white')
cd(Result_folder)
saveas(gcf,'Signal_ReconstructionLBFGS','fig')
cd(oldfolder)

predicted_states=X_proposed_lbfgs_unchangedGA;
%predicted_states=X_proposed_lbfgsGA;

 objReal=((double(Simm_GA_unchanged-y_train).^2)); 
objReal=(objReal).^(0.5);
obj=mean(objReal);
objStd=std(objReal);

objss.values=objReal;
objss.discomfort=obj;
objss.deviation=objStd;
case 2

disp('********** Block Iterative Ensemble Smoother Method**************')
%N=input('Enter size of the ensemble: ');
N=500;
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
methodInfo.maxInnerIter = 10; % maximum iteration number in the inner loop
methodInfo.init_lambda = 1; % initial lambda value
methodInfo.lambda_reduction_factor = 0.9; % reduction factor in case to reduce gamma
methodInfo.lambda_increment_factor = 2; % increment factor in case to increase gamma 
methodInfo.doTSVD = 1; % do a TSVD on the cov of simulated obs? (1 = yes)
obv_nLevel = 1;
if methodInfo.doTSVD
    methodInfo.tsvdCut = 0.99; % discard eigenvalues/eigenvectors if they are not among the truncated leading ones
end
methodInfo.min_RN_change = 1; % minimum residual norm (RN) change (in percentage); RN(k) - RN(k+1) > RN(k) * min_RN_change / 100
 ensemble = iES2(methodInfo,...
    y_train,hyppsini,clfx,clfy,sizec,oldfolder,Model_4,augment);
	
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
spit=abs((Forwarding2(aa,clfx,...
clfy,oldfolder,Model_4,augment)));
spitmean=reshape(spit,[],size(y_train,2));


aa=hyp_mean_unchanged;
spit=abs((Forwarding2(aa,clfx,...
clfy,oldfolder,Model_4,augment)));
spitmeanraw=reshape(spit,[],size(y_train,2));


figure()
plot(y_train(1:szss,1),'r','LineWidth',1)
hold on
% plot(spitmean,'k--','LineWidth',1)
% hold on
plot(spitmeanraw,'b--','LineWidth',1)
hold off
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Set point-temperature','FontName','Helvetica', 'Fontsize', 13);
title('Set point Temperature ','FontName','Helvetica', 'Fontsize', 13)
legend('Set point Temperature','Batch Controller:I-ES ','location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')

cd(Result_folder)
saveas(gcf,'Signal_ReconstructionIES','fig')
cd(oldfolder)

 objReal=(((double(spitmeanraw)-y_train).^2)); 
objReal=(objReal).^(0.5);
obj=mean(objReal);
objStd=std(objReal);

objss.values=objReal;
objss.discomfort=obj;
objss.deviation=objStd;
end
rmpath(ff);
disp('*******************PROGRAMME EXECUTED******************************')
end
