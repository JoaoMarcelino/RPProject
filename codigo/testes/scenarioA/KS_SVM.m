close all;
clear all;

matrix=[];
for i=1:5
    i
    rng(i);
    data=load('data.mat').data;
    [data_train,data_test]=splitDataset(data,200000);
    data_train=chooseScenario(data_train,1);
    data_test=chooseScenario(data_test,1);
    
    %kruskall
    data_train.X=data_train.X([4,7,9,11,13],:);
    data_test.X=data_test.X([4,7,9,11,13],:);
    data_train=resample(data_train,5000);
    data_test=resample(data_test,5000);
    
    [pred_y,true_y]=svmClassifier(data_train,data_test);
    [accuracy,specificity,sensibility,fscore]=computePerformance(pred_y,true_y);
    matrix=[matrix,[accuracy;sensibility;specificity;fscore]];
end
fprintf('Accuracy: %f (%s %f)\n',mean(matrix(1,:)),char(177),std(matrix(1,:)));
fprintf('Sensibility: %f (%s %f)\n',mean(matrix(2,:)),char(177),std(matrix(2,:)));
fprintf('Specificity: %f (%s %f)\n',mean(matrix(3,:)),char(177),std(matrix(3,:)));
fprintf('FScore: %f (%s %f)\n',mean(matrix(4,:)),char(177),std(matrix(4,:)));
