function [pred_y,true_y] = svmClassifier(data_train,data_test)
%SVMCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
nSamples=size(data_train.X,2);
model = fitcsvm(data_train.X',data_train.y','KernelFunction','rbf','BoxConstraint',4,'kernelScale',4,'Standardize',true);
[pred_y]=predict(model,data_test.X');
pred_y=pred_y';
true_y=data_test.y;
end

