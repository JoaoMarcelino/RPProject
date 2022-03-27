close all
%4.1 Data import
xls_data = readmatrix('..\dataset\heart_2020_cleaned.csv');

%xls_data = col_names(2:end,:);
%col_names = col_names(1,:);

test_lim = 200000;

data.X=xls_data(1:test_lim, 5:end)';
data.y=xls_data(1:test_lim, 1:4)';
data.dim=size(data.X,1);
data.num_data=size(data.X,2);
data.name='Heart Diseases dataset';
data.indep_names=["_BMI5"	"Smoking"	"AlcoholDrinking"	"Stroke"	"PhysicalHealth"	"MentalHealth"	"DiffWalking"	"Sex"	"AgeCategory"	"Race"	"Diabetic"	"PhysicalActivity"	"GenHealth"	"SleepTime"	"Asthma"];
data.dep_names=["CoronaryHeartDisease"	"MyocardialInfarction"	"KidneyDisease"	"SkinCancer"];

%4.2 Feature Selection and Reduction

%4.2.1 Scaling
data=scalestd(data);

figure('Name','Data Distribution');
for i=1:data.dim
    subplot(3,5,i);
    histogram(data.X(i,:),10);
    title(data.indep_names(i));
end

%4.2.1 PCA
model = pca(data.X);
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
%4.3 Experimental Analysis
%4.4 Pattern Recognition Methods

