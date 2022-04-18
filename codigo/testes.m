rng(2);
% PCA + Minimmum Dist Classifier
close all;
clear all;
%   Data import
data=load('data.mat').data;
[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;

data_train=scalestd(data_train);
data_test=scalestd(data_test);
model = pca(data_train.X);
eingenvalues=model.eigval;

figure('Name',"Eingenvalues");
subplot(1,2,1);
plot([1:size(eingenvalues)],eingenvalues,'o-');
title("Eingenvalues");

for i=1:size(eingenvalues)
    variance=sum(eingenvalues(1:i))/sum(eingenvalues)*100;
    variance_preserved(1,i)=variance;
end
subplot(1,2,2);
bar(variance_preserved);
title("Variance Preserved");


model = pca(data_train.X,4);
data_train.X=model.W'*data_train.X+model.b;
data_test.X=model.W'*data_test.X+model.b;

[pred_y,true_y]=mdClassifier(data_train,data_test);
figure;
C = confusionmat(true_y,pred_y);
confusionchart(C);
accuracy=sum(pred_y==true_y)/size(true_y,2);
disp(accuracy);

%{
rng(2);
% PCA + Fisher LDA Minimmum Dist Classifier
close all;
clear all;
%   Data import
data=load('data.mat').data;
[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;

data_train=scalestd(data_train);
data_test=scalestd(data_test);
model = pca(data_train.X);
eingenvalues=model.eigval;

figure('Name',"Eingenvalues");
subplot(1,2,1);
plot([1:size(eingenvalues)],eingenvalues,'o-');
title("Eingenvalues");

for i=1:size(eingenvalues)
    variance=sum(eingenvalues(1:i))/sum(eingenvalues)*100;
    variance_preserved(1,i)=variance;
end
subplot(1,2,2);
bar(variance_preserved);
title("Variance Preserved");


model = pca(data_train.X,4);
data_train.X=model.W'*data_train.X+model.b;
data_test.X=model.W'*data_test.X+model.b;
[data_train,data_test]=ldaFisher(data_train,data_test);
[pred_y,true_y]=mdClassifier(data_train,data_test);
figure;
C = confusionmat(true_y,pred_y);
confusionchart(C);
accuracy=sum(pred_y==true_y)/size(true_y,2);
disp(accuracy);
%}