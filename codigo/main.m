close all;
clear;
%4.1 Data import and Scenario Choosing
[data_train,data_test]=chooseScenario(200000, 1);

%data visualization
figure('Name','Data Distribution');
for i=1:data_train.dim
    subplot(3,5,i);
    histogram(data_train.X(i,:),10);
    title(data_train.indep_names(i));
end
figure('Name','Data Distribution per Class');
histogram(data_train.y,2);
xlabel('Class');
ylabel('Number of occurences');
title('Data Distribution per Class')

%4.2 Feature Selection and Reduction

%4.2.1 Scaling and PCA
data_train=scalestd(data_train);
data_test=scalestd(data_test);
model = pca(data_train.X);
eingenvalues=model.eigval;

figure('Name',"Eingenvalues");
subplot(1,2,1);
plot([1:size(eingenvalues)],eingenvalues,'o-');
title("Eingenvalues");

variance_preserved=zeros([1,size(eingenvalues)]);

feature_reduction = 0;
percentage = 90;

for i=1:size(eingenvalues)
    variance=sum(eingenvalues(1:i))/sum(eingenvalues)*100;
    variance_preserved(1,i)=variance;

    if variance >= percentage && feature_reduction == 0
        feature_reduction = i;
    end
end
subplot(1,2,2);
bar(variance_preserved);
title("Variance Preserved");

disp("Kaisa First Features: ");
disp(eingenvalues(eingenvalues>=1));
disp("Scree Test: ");
disp(eingenvalues(1:3));


%%4.2.2 Kruscal wallis and correlation coefficients matrix 
for i=1:data_train.dim
    [p,atab,stats]=kruskalwallis(data_train.X(i,:),data_train.y,'off');
    rank{i,1}=data_train.indep_names{i};
    rank{i,2}=atab{2,5};
    atab;
end
disp(rank);
C=corrcoef(data_train.X');
figure('Name',"Correlation Coefficients Matrix");
heatmap(C);
title("Correlation Coefficients Matrix");

%%4.2.3 Fisher LDA
%4.1 Data import
data=load('data.mat').data;
[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;

[data_train,data_test]=ldaFisher(data_train,data_test);

figure;
X1=data_train.X(data_train.y==0);
X2=data_train.X(data_train.y==1);
scatter(X1,(1:size(X1,2)).*0+0.25,'b');
hold on;
scatter(X2,(1:size(X2,2)).*0-0.25,'r');

%4.3 Minimum Distance classifier
[pred_y,true_y]=mdClassifier(data_train,data_test);



% k-Nearest Neighbor

data = load('data.mat').data;
data= scalestd(data);

[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;


n_runs = 2;
k= 10;
[pred_y,true_y] = knn(data_train, n_runs, k);



%%
%Bayes Classifier
data = load('data.mat').data;
[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;

costs=[0 1; 0.01 0];
[pred_y,true_y]=bayesClassifier(data_train,data_test,costs);
[accuracy,specificity,sensibility]=computePerformance(pred_y,true_y);

C = confusionmat(true_y,pred_y);
confusionchart(C,[0 1]);


%%
%SVM Classifier
data = load('data.mat').data;
[data_train,data_test]=splitDataset(data,200000);
data_train.y=data_train.y(1,:)-1;
data_test.y=data_test.y(1,:)-1;

data_train.X=data_train.X(1:4,:);
data_test.X=data_test.X(1:4,:);

constraint=1;
gamma=1;
%func='linear';
func='rbf';
%func='polynomial';
[pred_y,true_y]=svmClassifier(data_train,data_test,func,constraint,gamma);
[accuracy,specificity,sensibility]=computePerformance(pred_y,true_y);

C = confusionmat(true_y,pred_y);
confusionchart(C,[0 1]);