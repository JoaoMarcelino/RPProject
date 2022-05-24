function [pred_y,true_y] = bayesClassifierCost(data_train,data_test,costRatio)
%BAYESCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
Mdl=fitcnb(data_train.X',data_train.y');
Mdl.Cost=[0 1.*costRatio; 1 0];
pred_y = predict(Mdl,data_test.X');

pred_y=pred_y';
true_y=data_test.y;
end

