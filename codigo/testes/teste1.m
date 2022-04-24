% PCA + Fisher LDA + LDA
close all;
clear all;
rng(2);

%Data import
data=load('data.mat').data;
[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;

%PCA
numFeatures=4;
data_train=scalestd(data_train);
data_test=scalestd(data_test);

model = pca(data_train.X,numFeatures);
data_train.X=model.W'*data_train.X+model.b;
data_test.X=model.W'*data_test.X+model.b;

%Fisher LDA
[data_train,data_test]=ldaFisher(data_train,data_test);

%LDA
[pred_y,true_y]=mdClassifier(data_train,data_test);
figure;
C = confusionmat(true_y,pred_y);
confusionchart(C);
accuracy=sum(pred_y==true_y)/size(true_y,2);
error=cerror(pred_y,true_y);
disp(accuracy);
disp(error);
