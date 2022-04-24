% Kruskal + Mahal. Minimmum Dist Classifier
close all;
clear all;
matrix=[];
for i=1:30
    rng(i);
    data=load('data.mat').data;
    [data_train,data_test]=splitDataset(data,200000);
    data_train.y=data_train.y(1,:)-1;
    data_test.y=data_test.y(1,:)-1;
    
    %kruskall
    data_train.X=data_train.X([4,7,9,11,13],:);
    data_test.X=data_test.X([4,7,9,11,13],:);

    [pred_y,true_y]=mdClassifier(data_train,data_test);
    [accuracy,specificity,sensibility]=computePerformance(pred_y,true_y);
    matrix=[matrix,[accuracy;specificity;sensibility]];
end
save('test7.mat','matrix');
disp(mean(matrix(1,:)));
disp(std(matrix(1,:)));
disp(mean(matrix(3,:)));
disp(std(matrix(3,:)));
disp(mean(matrix(2,:)));
disp(std(matrix(2,:)));

