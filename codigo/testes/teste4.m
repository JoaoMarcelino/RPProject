close all;
clear all;
matrix=[];
for i=1:30
    rng(i);
    data=load('data.mat').data;
    [data_train,data_test]=splitDataset(data,200000);
    data_train.y=data_train.y(1,:)-1;
    data_test.y=data_test.y(1,:)-1;
    
    [pred_y,true_y]=mdClassifier(data_train,data_test);
    [accuracy,specificity,sensibility]=computePerformance(pred_y,true_y);
    matrix=[matrix,[accuracy;specificity;sensibility]];
end
save('test4.mat','matrix');