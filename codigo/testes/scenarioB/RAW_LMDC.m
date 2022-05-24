% Raw data + Linear Minimmum Dist Classifier
close all;
clear all;
matrix=[];
for i=1:30
    rng(i);
    data=load('data.mat').data;
    [data_train,data_test]=splitDataset(data,200000);
    data_train=chooseScenario(data_train,2);
    data_test=chooseScenario(data_test,2);
    
    [pred_y,true_y]=mdClassifier(data_train,data_test);
    [accuracy,specificity,sensibility,fscore]=computePerformance(pred_y,true_y);
    matrix=[matrix,[accuracy;sensibility;specificity;fscore]];
end

fprintf('Accuracy: %f (%s %f)\n',mean(matrix(1,:)),char(177),std(matrix(1,:)));
fprintf('Sensibility: %f (%s %f)\n',mean(matrix(2,:)),char(177),std(matrix(2,:)));
fprintf('Specificity: %f (%s %f)\n',mean(matrix(3,:)),char(177),std(matrix(3,:)));
fprintf('FScore: %f (%s %f)\n',mean(matrix(4,:)),char(177),std(matrix(4,:)));
