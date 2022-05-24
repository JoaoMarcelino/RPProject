close all;
clear all;
matrix=[];
for i=1:30
    i
    rng(i);
    data=load('data.mat').data;
    [data_train,data_test]=splitDataset(data,200000);
    data_train=chooseScenario(data_train,3);
    data_test=chooseScenario(data_test,3);
    
    %kruskall
    data_train.X=data_train.X([4,7,9,11,13],:);
    data_test.X=data_test.X([4,7,9,11,13],:);
    
    tmp = templateLinear();
    ecoc = fitcecoc(data_train.X,data_train.y,'Coding','onevsall','Learners',tmp,'ObservationsIn','columns');
    [pred_y]=predict(ecoc,data_test.X')';
    true_y=data_test.y;
    accuracy=sum(pred_y==true_y)/size(true_y,2);
    tp=sum(pred_y==1 & true_y==1);
    fn=sum(pred_y~=1 & true_y==1);
    sensitivity=tp/(tp+fn);
    matrix=[matrix,[accuracy;sensitivity]];
end
fprintf('Accuracy: %f (%s %f)\n',mean(matrix(1,:)),char(177),std(matrix(1,:)));
fprintf('Sensitivity: %f (%s %f)\n',mean(matrix(2,:)),char(177),std(matrix(2,:)));

