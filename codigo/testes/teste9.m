% Kruskal + Mahal. Minimmum Dist Classifier
close all;
clear all;

rng(1);
data=load('data.mat').data;
[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;

%kruskall
data_train.X=data_train.X([4,7,9,11,13],:);
data_test.X=data_test.X([4,7,9,11,13],:);

[pred_y,true_y]=mdClassifier(data_train,data_test);
[accuracy,specificity,sensibility]=computePerformance(pred_y,true_y);

C = confusionmat(true_y,pred_y);
confusionchart(C,[0 1]);

