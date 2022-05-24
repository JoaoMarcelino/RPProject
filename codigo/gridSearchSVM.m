close all;
clear all;
numFeatures=4;
matrix=[];
gamma=[-3:4];
cost=[-3:4];
gamma=2.^gamma;
cost=2.^cost;

valuesAcc = zeros([size(cost,2),size(gamma,2)]);
valuesSens = zeros([size(cost,2),size(gamma,2)]);
x=1;
for c=cost
    x
    y=1;
    for g=gamma
        matrix=[];
        for i=1:2
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
            
            model = fitcsvm(data_train.X',data_train.y','KernelFunction','rbf','BoxConstraint',c,'kernelScale',g,'Standardize',true);
            [pred_y]=predict(model,data_test.X')';
            true_y=data_test.y;
            [accuracy,specificity,sensibility,fscore]=computePerformance(pred_y,true_y);
            matrix=[matrix,[accuracy;sensibility;specificity;fscore]];
        end
        valuesAcc(x,y)=mean(matrix(1,:));
        valuesSens(x,y)=mean(matrix(2,:));
        y=y+1;
    end
    x=x+1;
end

%%
figure;
[X,Y] = meshgrid(cost,gamma);
contourf(X,Y,valuesSens);
title('Sensibility Grid Search');
xlabel("BoxConstraint");
ylabel("kernelScale");
colorbar

figure;
[X,Y] = meshgrid(cost,gamma);
contourf(X,Y,valuesAcc);
title('Accuracy Grid Search');
xlabel("BoxConstraint");
ylabel("kernelScale");
colorbar
