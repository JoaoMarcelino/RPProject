
close all;
clear all;
for k=[3,10,20]
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
        
        [pred_y,true_y]=knn(data_train,data_test,k);
        [accuracy,specificity,sensibility,fscore]=computePerformance(pred_y,true_y);
        matrix=[matrix,[accuracy;sensibility;specificity;fscore]];
    end
    fprintf('Accuracy: %f (%s %f)\n',mean(matrix(1,:)),char(177),std(matrix(1,:)));
    fprintf('Sensibility: %f (%s %f)\n',mean(matrix(2,:)),char(177),std(matrix(2,:)));
    fprintf('Specificity: %f (%s %f)\n',mean(matrix(3,:)),char(177),std(matrix(3,:)));
    fprintf('FScore: %f (%s %f)\n',mean(matrix(4,:)),char(177),std(matrix(4,:)));
end
