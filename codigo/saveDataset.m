xls_data = readmatrix('..\dataset\heart_2020_cleaned.csv');

%Total Data
data.X=xls_data(:, 5:end)';
data.y=xls_data(:, 1:4)';
data.dim=size(data.X,1);
data.num_data=size(data.X,2);
data.name='Heart Diseases dataset';
data.indep_names=["_BMI5"	"Smoking"	"AlcoholDrinking"	"Stroke"	"PhysicalHealth"	"MentalHealth"	"DiffWalking"	"Sex"	"AgeCategory"	"Race"	"Diabetic"	"PhysicalActivity"	"GenHealth"	"SleepTime"	"Asthma"];
data.dep_names=["CoronaryHeartDisease"	"MyocardialInfarction"	"KidneyDisease"	"SkinCancer"];

save('data.mat','data');