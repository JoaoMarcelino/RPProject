function [pred_y,true_y] = svmClassifier(data_train,data_test)
%SVMCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
model = fitcsvm(data_train.X',data_train.y','KernelFunction','rbf','BoxConstraint',4,'kernelScale',4,'Standardize',true);
[pred_y]=predict(model,data_test.X')';
true_y=data_test.y;
end

