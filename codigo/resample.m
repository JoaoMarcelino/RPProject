function [data] = resample(data,nSamples)
%RESAMPLE Summary of this function goes here
%   Detailed explanation goes here
data_X1=data.X(:,find(data.y==1));
data_X2=data.X(:,find(data.y==2));

ix = randperm(size(data_X1,2));
data_X1_ix = ix(1:nSamples);
data_X2_ix = ix(1:nSamples);
%Training set
data_X1 = data_X1(:,data_X1_ix);
data_X2 = data_X2(:,data_X2_ix);

data.X=[data_X1 data_X2];
data.y=[ones([1,size(data_X1,2)]) ones([1,size(data_X2,2)])+1];
data.num_data=size(data.X,2);
end

