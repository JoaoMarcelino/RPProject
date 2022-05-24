close all;
clear all;
clc;
%data visualization
%Features Histograms
%%
data=load('data.mat').data;
figure('Name','Data Distribution');
for i=1:data.dim
    subplot(3,5,i);
    histogram(data.X(i,:),10);
    title(data.indep_names(i));
end
%%
%Class Distribution
for i=1:3
    data=load('data.mat').data;
    data=chooseScenario(data,i);
    fig=figure('Name','Data Distribution per Class');
    y1=sum(data.y==1);
    y2=sum(data.y==2);
    y3=sum(data.y==3);
    bar([y1,y2,y3]);
    xlabel('Class');
    ylabel('Number of occurences');
    title(sprintf('Data Distribution per Class - Scenario %d',i));
end


%%4.2.2 Kruscal wallis and correlation coefficients matrix
%%
%scenario A
data=load('data.mat').data;
data=chooseScenario(data,1);
for i=1:data.dim
    [p,atab,stats]=kruskalwallis(data.X(i,:),data.y,'off');
    rank{i,1}=data.indep_names{i};
    rank{i,2}=sprintf('%.7f',atab{2,5});
end
disp(rank);
C=corrcoef(data.X');
figure('Name',"Correlation Coefficients Matrix");
heatmap(round(C,2));
title("Correlation Coefficients Matrix");

%scenario B
data=load('data.mat').data;
data=chooseScenario(data,2);
for i=1:data.dim
    [p,atab,stats]=kruskalwallis(data.X(i,:),data.y,'off');
    rank{i,1}=data.indep_names{i};
    rank{i,2}=sprintf('%.7f',atab{2,5});
end
disp(rank);

%scenario C
data=load('data.mat').data;
data=chooseScenario(data,3);
for i=1:data.dim
    [p,atab,stats]=kruskalwallis(data.X(i,:),data.y,'off');
    rank{i,1}=data.indep_names{i};
    rank{i,2}=sprintf('%.7f',atab{2,5});
end
disp(rank);
%%
%PCA
data=load('data.mat').data;
data=chooseScenario(data,1);
data=scalestd(data);
model = pca(data.X);
eingenvalues=model.eigval;

fig=figure('Name',"Eingenvalues");
subplot(1,2,1);
plot([1:size(eingenvalues)],eingenvalues,'o-');
xlabel("Component");
ylabel("Eingenvalues");

for i=1:size(eingenvalues)
    variance=sum(eingenvalues(1:i))/sum(eingenvalues)*100;
    variance_preserved(1,i)=variance;
end
subplot(1,2,2);
bar(variance_preserved);
sgtitle('Principal Component Analysis');
xlabel('Number of Components');
ylabel("Variance Preserved (cumulative)");

%%
%Age Category per class
data=load('data.mat').data;
data=chooseScenario(data,1);
figure;
histogram(data.X(9,find(data.y==1)),'Normalization','probability','FaceAlpha',0.70);
hold on;
histogram(data.X(9,find(data.y==2)),'Normalization','probability','FaceAlpha',0.70);
legend('Class 1 (positive)','Class 2 (negative)');
title('Data Distribution per Class According to Age Category -  Scenario 1');
xlabel('Age Category');
ylabel('Relative Frequency');

data=load('data.mat').data;
data=chooseScenario(data,1);
[data,data]=ldaFisher(data,data);
figure;
histogram(data.X(:,find(data.y==1)),33,'Normalization','probability','FaceAlpha',0.70);
hold on;
histogram(data.X(:,find(data.y==2)),33,'Normalization','probability','FaceAlpha',0.70);
legend('Class 1 (positive)','Class 2 (negative)');
title('Data Distribution per Class after Fisher'' LDA -  Scenario 1');
xlabel('Age Category');
ylabel('Relative Frequency');


%%
%effect of scaling data
data=load('data.mat').data;
data=chooseScenario(data,1);
X1=data.X(9,:);
X2=data.X(13,:);
figure;
boxplot([X1,X2],[zeros(1,size(X1,2)),ones(1,size(X2,2))],'Labels',{'Age','General Health'});
title("Before Scaling (Age and General Health Boxplots)");

data=load('data.mat').data;
data.y=data.y(1,:)-1;
data=scalestd(data);
X1=data.X(9,:);
X2=data.X(13,:);
figure;
boxplot([X1,X2],[zeros(1,size(X1,2)),ones(1,size(X2,2))],'Labels',{'Age','General Health'});
title("After Scaling (Age and General Health Boxplots)");

%%
%ROC CURVES
clear all;
close all;
data=load('data.mat').data;
data=chooseScenario(data,1);
figure('Name','Data Distribution');
for i=1:data.dim
    [FPR,FNR]=roc(data.X(i,:),data.y);
    subplot(3,5,i);
    plot(FPR,1-FNR);
    title(data.indep_names(i));
    xlabel('FPRate');
    ylabel('TPR');
end

clear all;
close all;
for i=[1:3]
    data=load('data.mat').data;  
    data=chooseScenario(data,i);
    for k=1:data.dim
        [FPR,FNR]=roc(data.X(k,:),data.y);
        AUC=rocArea(FPR,FNR);
        rank{k,1}=data.indep_names{k};
        rank{k,2}=sprintf('%.7f',AUC);
    end
    disp(rank);
end

