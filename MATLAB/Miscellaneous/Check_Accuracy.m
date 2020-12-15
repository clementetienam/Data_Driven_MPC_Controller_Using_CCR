function [costtrain,costtest,Truee,Valueehard]=Check_Accuracy...
    (X_train,X_test,y_train,...
    y_test,clfy,machine,oldfolder,foldera,name,ind_train,ind_test)

% ind_train=sort(ind_train);
% ind_test=sort(ind_test);
jj=[ind_train(:);ind_test(:)];
jj=numel(jj);

ytrainpred=reshape((predict(machine,X_train'))',[],size(y_train,2));
ytrainpred=clfy.inverse_transform(ytrainpred);
y_train=clfy.inverse_transform(y_train);

ytestpred=reshape((predict(machine,X_test'))',[],size(y_train,2));
ytestpred=clfy.inverse_transform(ytestpred);
y_test=clfy.inverse_transform(y_test);


Valueehard=zeros(jj,size(y_train,2));
Valueehard(ind_train,:)=ytrainpred;
Valueehard(ind_test,:)=ytestpred;

Truee=zeros(jj,size(y_train,2));
Truee(ind_train,:)=y_train;
Truee(ind_test,:)=y_test;


disp('-----------------------------Check Training accuracy----------------')
CCR=ytrainpred;
True=y_train;
yesup=sum((abs(CCR-True)).^2);
yesdown=sum((True-mean(True,1)).^2);
R2=1-(yesup./yesdown);
R2=R2*100;
yesdown2=sum((abs(True)).^2);
L2=1-((yesup/yesdown2).^0.5);
L2=L2*100;
matrix=True-CCR;
L1norm=norm(matrix,1);
L2norm=norm(matrix,2);
absolute_error=abs(matrix);
MAE=mean(absolute_error,1);
squared_error=(absolute_error).^2;
MSE=mean(squared_error);
RMSE=MSE.^0.5;
R2train=R2;

costtrain=cell(6,2);
costtrain{1,1}='R2';
costtrain{2,1}='L2';
costtrain{3,1}='L1norm';
costtrain{4,1}='L2norm';
costtrain{5,1}='MAE';
costtrain{6,1}='RMSE';

costtrain{1,2}=R2;
costtrain{2,2}=L2;
costtrain{3,2}=L1norm;
costtrain{4,2}=L2norm;
costtrain{5,2}=MAE;
costtrain{6,2}=RMSE;

fprintf('The R2 accuracy for prediction on training data is %4.2f \n',R2); 
fprintf('The L2 accuracy for prediction on training data is %4.2f \n',L2);
fprintf('The RMSE for prediction on training data is %4.2f \n',RMSE); 

disp('-----------------------------Check Test accuracy----------------')
CCR=ytestpred;
True=y_test;
yesup=sum((abs(CCR-True)).^2);
yesdown=sum((True-mean(True,1)).^2);
R2=1-(yesup/yesdown);
R2=R2*100;
yesdown2=sum((abs(True)).^2);
L2=1-((yesup/yesdown2).^0.5);
L2=L2*100;
matrix=True-CCR;
L1norm=norm(matrix,1);
L2norm=norm(matrix,2);
absolute_error=abs(matrix);
MAE=mean(absolute_error,1);
squared_error=(absolute_error).^2;
MSE=mean(squared_error);
RMSE=MSE.^0.5;
R2test=R2;
costtest=cell(6,2);
costtest{1,1}='R2';
costtest{2,1}='L2';
costtest{3,1}='L1norm';
costtest{4,1}='L2norm';
costtest{5,1}='MAE';
costtest{6,1}='RMSE';

costtest{1,2}=R2;
costtest{2,2}=L2;
costtest{3,2}=L1norm;
costtest{4,2}=L2norm;
costtest{5,2}=MAE;
costtest{6,2}=RMSE;

fprintf('The R2 accuracy for prediction on test data is %4.2f \n',R2); 
fprintf('The L2 accuracy for prediction on test data is %4.2f \n',L2);
fprintf('The RMSE for prediction on test data is %4.2f \n',RMSE);


if size(y_train,2)==1
disp('---------------------Plot Figures Now------------------------------')
figure()
subplot(2,3,1)
line(ytrainpred,y_train,'Tag','Data','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[1 0 0],...
    'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 1]);
title('(a)-Training accuracy','FontName','Helvetica', 'Fontsize', 10);
shading flat
grid off
colormap('jet')
xlabel('Machine','FontSize',10,'FontName','Helvetica');
ylabel('True','FontSize',10,'FontName','Helvetica');     
line([min([ytrainpred,y_train]),max([ytrainpred,y_train])],...
    [min([ytrainpred,y_train]),max([ytrainpred,y_train])],'Tag','Reference Ends','LineWidth',3,'color','black');
str=['R2 = ',num2str(R2train)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 10, 'verticalalignment', 'top', 'horizontalalignment', 'left');

set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,2)
hist(ytrainpred-y_train)
shading flat
grid off
title('(b)-Dissimilarity(Training)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,4)
line(ytestpred,y_test,'Tag','Data','MarkerFaceColor',[1 0 0],...
    'MarkerEdgeColor',[1 0 0],...
    'Marker','o',...
    'LineStyle','none',...
    'Color',[0 0 1]);
title('(d)-Testing','FontName','Helvetica', 'Fontsize', 10);
shading flat
grid off
colormap('jet')
xlabel('Machine','FontSize',10,'FontName','Helvetica');
ylabel('True','FontSize',10,'FontName','Helvetica');     
line([min([ytestpred,y_test]),max([ytestpred,y_test])],...
    [min([ytestpred,y_test]),max([ytestpred,y_test])],'Tag','Reference Ends','LineWidth',3,'color','black');
str=['R2 = ',num2str(R2test)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 10, 'verticalalignment', 'top', 'horizontalalignment', 'left');

set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')

subplot(2,3,5)
hist(ytestpred-y_test)
shading flat
grid off
title('(e)-Dissimilarity(Test)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Count', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('Difference', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')


subplot(2,3,3)
plot(Truee,'+r');
hold on
plot(Valueehard,'k','LineWidth', 1)

shading flat
grid off
title('(c)-Machine Reconstruction(Hard-Prediction)','FontName','Helvetica', 'Fontsize', 10);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 10);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 10);
colormap('jet')
set(gca, 'FontName','Helvetica', 'Fontsize', 10)
set(gcf,'color','white')
h = legend('True[train+test]','Machine[train+test]');
set(h,'FontSize',10);
end



cd(foldera)
%Namefilef=strcat('performance_', sprintf('%d',jjm),'.fig');
Namefilef=name;
saveas(gcf,Namefilef)
cd(oldfolder)
end