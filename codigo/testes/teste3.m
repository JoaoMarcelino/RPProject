% PCA + Mahal. MDC
close all;
clear all;

for numFeatures=[1,4,12]
    matrix=[];
    for i=1:30
        rng(i);
        %   Data import
        data=load('data.mat').data;
        [data_train,data_test]=splitDataset(data,200000);
        data_train.y=data_train.y(1,:)-1;
        data_test.y=data_test.y(1,:)-1;
        
        
        %PCA
        data_train=scalestd(data_train);
        data_test=scalestd(data_test);
        
        model = pca(data_train.X,numFeatures);
        data_train.X=model.W'*data_train.X+model.b;
        data_test.X=model.W'*data_test.X+model.b;

        %mdc
        [pred_y,true_y]=mahalClassifier(data_train,data_test);
        [accuracy,specificity,sensibility]=computePerformance(pred_y,true_y);
        matrix=[matrix,[accuracy;specificity;sensibility]];
    end
    disp(mean(matrix(1,:)));
    disp(std(matrix(1,:)));
    disp(mean(matrix(3,:)));
    disp(std(matrix(3,:)));
    disp(mean(matrix(2,:)));
    disp(std(matrix(2,:)));
    test3{numFeatures}=matrix;
end
save('test3.mat','test3');