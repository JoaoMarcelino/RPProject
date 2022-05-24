function [pred_y,true_y] = bayesClassifier(data_train,data_test)
%BAYESCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
Mdl=fitcnb(data_train.X',data_train.y');
Mdl.Cost=[0 25; 1 0];
pred_y = predict(Mdl,data_test.X');

pred_y=pred_y';
true_y=data_test.y;
end

