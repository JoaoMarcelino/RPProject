close all;
clear all;
%4.1 Data import
data=load('data.mat').data;

%scenario A - CoronaryHeartDisease: 0/1
data.y=data.y(1,:)-1;

data=scalestd(data);
%data visualization
figure('Name','Data Distribution');
for i=1:data.dim
    subplot(3,5,i);
    histogram(data.X(i,:),10);
    title(data.indep_names(i));
end
figure('Name','Data Distribution per Class');
histogram(data.y,2);
xlabel('Class');
ylabel('Number of occurences');
title('Data Distribution per Class')

%%4.2.2 Kruscal wallis and correlation coefficients matrix
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

figure;
histogram(data.X(14,data.y==0),20,'Normalization','probability');
hold on;
histogram(data.X(14,data.y==1),20,'Normalization','probability');