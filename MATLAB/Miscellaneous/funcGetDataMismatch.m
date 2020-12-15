function [obj,objStd,objReal]=funcGetDataMismatch(simData,measurment)
%: function [obj,objStd,objReal]=funcGetDataMismatch(simData,W,measurment)
%: compute the mismatch between simData and measurement;

ne=size(simData,2);
objReal=zeros(ne,1);
for j=1:ne
    objReal(j)=sum(((double(simData(:,j))-measurment).^2)); % for diagonal weight matrices only
end
 
obj=mean(objReal);
objStd=std(objReal);