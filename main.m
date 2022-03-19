
%4.1 Data import
[xls_data,col_names]=xlsread('heart_2020_cleaned.csv','heart_2020_cleaned' );

xls_data = col_names(2:end,:);
col_names = col_names(1,:);

test_lim = 200000;

data.X=xls_data(1:test_lim, 5:end)';
data.y=xls_data(1:test_lim, 1:4)';
data.dim=size(data.X,1);
data.num_data=size(data.X,2);
data.name='Heart Diseases dataset';

%4.2 Feature Selection and Reduction
%4.3 Experimental Analysis
%4.4 Pattern Recognition Methods