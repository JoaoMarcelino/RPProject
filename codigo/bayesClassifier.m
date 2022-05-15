function [pred_y,true_y] = bayesClassifier(data_train,data_test,costs)
%BAYESCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
Mdl=fitcnb(data_train.X',data_train.y');
Mdl.Cost=costs;
pred_y = predict(Mdl,data_test.X');

pred_y=pred_y';
true_y=data_test.y;
end

