close all;
clear all;

data=load('data.mat').data;
data.y=data.y(1,:)-1;

[data,data]=ldaFisher(data,data);
X1=data.X(1,find(data.y==0));
X2=data.X(1,find(data.y==1));
figure;
H1=histogram(X1,40,'Normalization','probability','FaceAlpha',0.70);
hold on;
H2=histogram(X2,40,'Normalization','probability','FaceAlpha',0.70);
hold off;
legend('Class 0 (positive)','Class 1 (negative)');
ylabel('Relative Probability');
xlabel('Feature Value');