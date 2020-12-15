function [Ynew,aj] = EnKF (sgsim,f, N, Sim1)

 sgsim = reshape(sgsim,[],N);


%disp(' determine the standard deviation for measurement pertubation')
parfor i=1:size(f,1)
stddoil(i,:)=0.01*f(i,:);
end


%disp('  generate Gaussian noise for the observed measurments  ');

Error1=ones(size(f,1),N);
parfor i=1:size(f,1)
Error1(i,:)=normrnd(0,stddoil(i,:),1,N);
end

Cd2 = (Error1*Error1')./(N-1);


parfor i=1:N
    Dj(:,i)=f+(Error1(:,i));	
 end

%disp('  generate the ensemble state matrix containing parameters and states  ');

overall=zeros(size(sgsim,1)+size(f,1),N); 

overall(1:size(sgsim,1),1:N)=sgsim;
overall(size(sgsim,1)+1:end,1:N)=Sim1;
Y=overall; %State variable,it is important we include simulated measurements in the ensemble state variable
%Y=[Y;Sim1];
M = mean(Sim1,2);
% Mean of the ensemble state
M2=mean(overall,2);
%M=M'
% Get the ensemble states pertubations
parfor j=1:N
    S(:,j)=Sim1(:,j)-M;
end
parfor j=1:N
    yprime(:,j)=overall(:,j)-M2;
end

H=zeros(size(f,1),size(overall,1));
H(:,end-size(f,1)+1:end)=eye(size(f,1));
%disp('  update the new ensemble  ');
unie=H*yprime;
Sim=H*Y;
unie2=unie+Error1;
[U0,Z0,V0] = svd(unie2,'econ');
joy2=Z0*Z0';
X1=pinv(joy2)*U0';
% Residual vector
X2=X1*(Dj-Sim);

X3=U0*X2;

X4=unie'*X3;
%Update the ensemble state
Ynew1=Y+(yprime*X4);
aj=Ynew1(end,:);
Ynew1(end,:)=[];
Ynew=Ynew1;
end