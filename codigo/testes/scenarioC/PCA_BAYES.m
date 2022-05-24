close all;
clear all;
numFeatures=4;
matrix=[];
for i=1:10
    i
    rng(i);
    data=load('data.mat').data;
    [data_train,data_test]=splitDataset(data,20000);
    data_train=chooseScenario(data_train,3);
    data_test=chooseScenario(data_test,3);
    
    %PCA
    data_train=scalestd(data_train);
    data_test=scalestd(data_test);
    
    model = pca(data_train.X,numFeatures);
    data_train.X=model.W'*data_train.X+model.b;
    data_test.X=model.W'*data_test.X+model.b;

    Mdl = templateNaiveBayes('Cost',[0 100; 1 0]);
    %Mdl.Cost=[0 25 25; 1 0 1; 1 1 0];
    ecoc = fitcecoc(data_train.X',data_train.y','Coding','onevsall','Learners',Mdl);
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
