close all;
clear all;
numFeatures=4;
matrix=[];
ratio=[1:5:100];

valuesAcc = zeros([1,size(ratio,2)]);
valuesSens = zeros([1,size(ratio,2)]);
x=1;
for r=ratio
    r
    matrix=[];
    for i=1:10
        rng(i);
        data=load('data.mat').data;
        [data_train,data_test]=splitDataset(data,200000);
        data_train=chooseScenario(data_train,1);
        data_test=chooseScenario(data_test,1);
        
        %PCA
        data_train=scalestd(data_train);
        data_test=scalestd(data_test);
        
        model = pca(data_train.X,numFeatures);
        data_train.X=model.W'*data_train.X+model.b;
        data_test.X=model.W'*data_test.X+model.b;
    
        data_train=resample(data_train,5000);
        data_test=resample(data_test,5000);
        
        Mdl=fitcnb(data_train.X',data_train.y');
        Mdl.Cost=[0 1*r; 1 0];
        pred_y = predict(Mdl,data_test.X')';
        true_y=data_test.y;
        [accuracy,specificity,sensibility,fscore]=computePerformance(pred_y,true_y);
        matrix=[matrix,[accuracy;sensibility;specificity;fscore]];
    end
    valuesAcc(x)=mean(matrix(1,:));
    valuesSens(x)=mean(matrix(2,:));
    x=x+1;
end

%%
figure;
plot(ratio,valuesAcc,'b');
hold on;
plot(ratio,valuesSens,'r');
legend('Accuracy','Sensibility');
xlabel('Cost Ratio');