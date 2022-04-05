close all;
clear;
%4.1 Data import
xls_data = readmatrix('..\dataset\heart_2020_cleaned.csv');

%Total Data
data.X=xls_data(:, 5:end)';
data.y=xls_data(:, 1:4)';
data.dim=size(data.X,1);
data.num_data=size(data.X,2);
data.name='Heart Diseases dataset';
data.indep_names=["_BMI5"	"Smoking"	"AlcoholDrinking"	"Stroke"	"PhysicalHealth"	"MentalHealth"	"DiffWalking"	"Sex"	"AgeCategory"	"Race"	"Diabetic"	"PhysicalActivity"	"GenHealth"	"SleepTime"	"Asthma"];
data.dep_names=["CoronaryHeartDisease"	"MyocardialInfarction"	"KidneyDisease"	"SkinCancer"];

%Random Perm
ix = randperm(data.num_data);

test_lim = 200000;

x_training = ix(1:floor(test_lim));
x_testing = ix(floor(test_lim):end);

%Training set
data_train.X = data.X(:,x_training);
data_train.y = data.y(x_training);
data_train.dim = size(data_train.X,1);
data_train.num_data = size(data_train.X,2);
data_train.name = "training Dataset";
data_train.indep_names = data.indep_names;
data_train.dep_names = data.dep_names;


%Testing set
data_test.X = data.X(:,x_testing);
data_test.y = data.y(x_testing);
data_test.dim = size(data_test.X,1);
data_test.num_data = size(data_test.X,2);
data_test.name = "testing Dataset";
data_test.indep_names = data.indep_names;
data_test.dep_names = data.dep_names;


%scenario A CoronaryHeartDisease:0/1
data_train.y=data_train.y(1,:)-1;

%4.2 Feature Selection and Reduction

%4.2.1 Scaling
data=scalestd(data_train);

%data visualization
figure('Name','Data Distribution');
for i=1:data_train.dim
    subplot(3,5,i);
    histogram(data_train.X(i,:),10);
    title(data_train.indep_names(i));
end

figure('Name','Data Distribution per Class');
for i=1:data_train.dim
    subplot(3,5,i);
    boxplot(data_train.X(i,:),data_train.y);
    title(data_train.indep_names(i));
end


%4.2.1 PCA
model = pca(data_train.X);
eingenvalues=model.eigval;

figure('Name',"Eingenvalues");
subplot(1,2,1);
plot([1:size(eingenvalues)],eingenvalues,'o-');
title("Eingenvalues");

variance_preserved=zeros([1,size(eingenvalues)]);
for i=1:size(eingenvalues)
    variance=sum(eingenvalues(1:i))/sum(eingenvalues)*100;
    variance_preserved(1,i)=variance;
end
subplot(1,2,2);
bar(variance_preserved);
title("Variance Preserved");

disp("Kaisa First Features: " + kaisa1);
disp("Kaisa Features with value greater than 1: " +  kaisa2);

%Kruscal wallis and correlation coefficients matrix 
for i=1:data_train.dim
    [p,atab,stats]=kruskalwallis(data_train.X(i,:),data_train.y,'off');
    rank{i,1}=data_train.indep_names{i};
    rank{i,2}=atab{2,5};
    atab;
end

C=corrcoef(data_train.X');
figure('Name',"Correlation Coefficients Matrix");
heatmap(C);
title("Correlation Coefficients Matrix");
%4.3 Experimental Analysis
%4.4 Pattern Recognition Methods
