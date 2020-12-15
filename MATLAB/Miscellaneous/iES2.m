function ensemble = iES2(methodInfo,...
    y_train,iniensemble,clfx,clfy,suni,oldfolder,model,augment)
% output = iES(modelInfo,obvInfo,methodInfo) 
% This scipt implements the iterative ensemble smoother (iES) in the
% paper "Iterative ensemble smoother as an approximate solution to a regularized 
% minimum-average-cost problem: theory and applications", SPE J. 2015.
% https://www.onepetro.org/journal-paper/SPE-176023-PA
%
% -- INPUTS --
% 
%
% The scipt "iES.m" implements the iterative ensemble smoother (iES) in the
% paper "Iterative ensemble smoother as an approximate solution to a regularized 
% minimum-average-cost problem: theory and applications", SPE J. 2015.
% https://www.onepetro.org/journal-paper/SPE-176023-PA
%
% The implementation here follows the paper "Levenberg–Marquardt forms of the
% iterative ensemble smoother for efficient history matching and uncertainty 
% quantification" by Chen and Oliver, Computational Geosciences, 2013. 
% 
% 
% By Dr Clement Etienam, PhD Petroleum Engineering 2018
%
%

%%
% record time and activate diary
ticID = tic;
%--  initialization  --%
iter = 0;
lambda = methodInfo.init_lambda;
isDisplay = 1; % display iteration information or not? (1=yes)
measurement=y_train;
nd = length(measurement); % dim of obs

%
ensemble = iniensemble;
ne = size(ensemble,2); % ensemble size
ensemble = [ensemble mean(ensemble,2)]; % note that ensemble mean is appended to the ensemble at the end
%--
R=eye(length(measurement));
[U,S,V] = svd(R);
obvInfo.principal_sqrtR = U * sqrt(S) * (V');
obvInfo.principal_sqrtR_inv = U * sqrt(S^(-1)) * (V');
perturbedData=zeros(nd,ne); % peturbations to observations
W = ones(size(measurement));
for j=1 : ne
    perturbedData(:,j) = measurement + W.*randn(size(measurement)); % normalized perturbations 
end

% simulated observations of the ensemble members 
nm = size(ensemble,1); % dim of system state  
% simData = [];
% for i = 1 : size(ensemble,2)
%     tmpData = funcGetSimData(modelInfo,obvInfo,ensemble(:,i));
%     simData = [simData reshape(tmpData,numel(tmpData),1)]; %#ok<*AGROW>
%     clear tmpData;
% end

simData= [];
for i=1:size(ensemble,2)
    aa=(ensemble(:,i));
	aa=reshape(aa,[],suni);
	spit=((Forwarding2(aa,clfx,clfy,oldfolder,model,augment)));
	spit=reshape(spit,[],1);
simData(:,i)=spit;
end

% data mismatch w.r.t the ensemble 
[obj,objStd,objReal]=funcGetDataMismatch(simData(:,1:ne),measurement);
init_obj = obj;
init_objStd = objStd;

% load information from methodInfo to configure iES
beta = methodInfo.beta;
objThreshold = beta^2 * nd;
% myDisplay(['   objThreshold=',num2str(objThreshold)],isDisplay);
maxOuterIter = methodInfo.maxOuterIter;
maxInnerIter = methodInfo.maxInnerIter;
init_lambda = methodInfo.init_lambda;
lambda_reduction_factor = methodInfo.lambda_reduction_factor;
lambda_increment_factor = methodInfo.lambda_increment_factor;
if methodInfo.doTSVD
    tsvdCut = methodInfo.tsvdCut;
end
min_RN_change = methodInfo.min_RN_change;
isTooSmallRNChange = 0;
exitFlag = [0 0 0]; % flags of iES termination status; 1st => maxOuterIter; 2nd => objThreshold; 3rd => min_RN_change

%-------------%
%-- run iES --%
%-------------%

% outer iteration loops 
while (iter < maxOuterIter) && (obj > objThreshold) 
    
    %--
    %myDisplay(['-- Outer iteration step: ' int2str(iter) ' --'],isDisplay);
    deltaM = ensemble(:,1:ne) - ensemble(:,ne+1) * ones(1,ne); %#ok<*NASGU>
    
    % deviation of the simulated observations 
    deltaD = simData(:,1:ne) - simData(:,ne+1)*ones(1,ne);

    %-- if do TSVD --%
    if methodInfo.doTSVD
        [Ud,Wd,Vd]=svd(deltaD,'econ');
        
        val=diag(Wd);
        total=sum(val);
        for j=1:ne
            svdPd=j;
            if (sum(val(1:j))/total > tsvdCut)
                break
            end
        end

        Vd=Vd(:,1:svdPd);
        Ud=Ud(:,1:svdPd);
        Wd=val(1:svdPd); % a vector
        clear val;
    end
    
    
    %-- initialization of the inner loop --%
    iterLambda=1;
    
    %-- inner iteration loop --%
    while iterLambda < maxInnerIter
        
        %myDisplay(['    -- Inner iteration step: ' int2str(iterLambda) '--'],isDisplay);
        
        ensembleOld = ensemble; % keep a copy of old status
        simDataOld = simData;
        
        if methodInfo.doTSVD
            alpha = lambda * sum(Wd.^2) / svdPd;
            %alpha = lambda * sum(Wd) / svdPd; % alternative rule 
            x1 = Vd * spdiags( Wd ./ (Wd.^2 + alpha), 0, svdPd, svdPd) ;
            KGain=deltaM*x1*Ud'; % Kalman gain
        else
            alpha = lambda * sum(sum(deltaD.^2)) / nd; % sum(sum(deltaD.^2)) = trace (deltaD * deltaD')
            KGain = deltaM * deltaD / (deltaD * deltaD' + alpha * eye(nd));
        end
        
        iterated_ensemble = ensemble(:,1:ne) - KGain * (simData(:,1:ne) - perturbedData);
        ensemble = [iterated_ensemble, mean(iterated_ensemble,2)];
        
        % check the change of ensemble mean 
        changeM = sqrt( sum( ( ensemble(:,ne+1) - ensembleOld(:,ne+1) ).^2 ) / nm );
       % myDisplay(['        average change (in RMSE) of the ensemble mean = ',num2str(changeM)],isDisplay);

        simData = [];
        for i=1:size(ensemble,2)
            aa=(ensemble(:,i));
            aa=reshape(aa,[],suni);
            spit=((Forwarding2(aa,clfx,clfy,oldfolder,model,augment)));
            spit=reshape(spit,[],1);
            simData(:,i)=spit;
        end

        [objNew,objStdNew,objRealNew]=funcGetDataMismatch(simData(:,1:ne),measurement);

              tmp_objReal = objReal;
              objReal = objRealNew;
              objReal = tmp_objReal;
              clear tmp_objReal;

        
        %--
        if objNew>obj
            lambda = lambda * lambda_increment_factor;
            %myDisplay(['         increasing Lambda to ',num2str(lambda)],isDisplay);
            iterLambda = iterLambda + 1;
            simData    = simDataOld;
            ensemble   = ensembleOld;
        else
            changeStd=(objStdNew-objStd)/objStd;
            %myDisplay(['        changeStd=',num2str(changeStd)],isDisplay);
            
            lambda = lambda * lambda_reduction_factor;%
            %myDisplay(['        reducing Lambda to ',num2str(lambda)],isDisplay);
            
            iter=iter+1;
            
            simDataOld=simData;
            ensembleOld=ensemble;
            objStd=objStdNew;
            obj=objNew;
            objReal = objRealNew;
            break % break the inner loop over lambda
        end
    end % end inner loop

    % if a better update not successfully found  
    
    if iterLambda >= maxInnerIter
        
        lambda = lambda * lambda_increment_factor;
        if lambda < init_lambda
            lambda = init_lambda;
        end


        
        iter=iter+1;

        %--
%         tline = '       terminating inner iterations: iterLambda >= maxInnerIter';
%         myDisplay(tline,isDisplay);
    end
    
    
    
    if isTooSmallRNChange 
        exitFlag(3) = 1;
        break
    end
end

if iter >= maxOuterIter    
%     tline = '   terminating outer iterations: iter >= maxOuterIter';
%     myDisplay(tline,isDisplay);
    exitFlag(1) = 1;
end
if obj <= objThreshold
%     tline = '   terminating outer iterations: obj <= objThreshold';
%     myDisplay(tline,isDisplay);
    exitFlag(2) = 1;
end

end