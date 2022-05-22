function [pred_y, test_y] = knn(data, n_runs, k)
%KNN Summary of this function goes here
%   Detailed explanation goes here
err_matrix = [];
err = [];
models_matrix = [];
models = [];

for i=1:data.dim
    [p,atab,stats]=kruskalwallis(data.X(i,:),data.y,'off');
    rank{i,1}=data.indep_names{i};
    rank{i,2}=atab{2,5};
    atab;
end

[Y, I] = sort([rank{:,2}], 2, 'descend');

%disp(rank(I) + "-" +  Y);

for i=1:n_runs
    ix = randperm(data.num_data);
    split = data.num_data * 0.5;
    
    %TRAIN
    data_train.X = data.X([I(1), I(2)],ix(1:split));
    data_train.y = data.y(:,ix(1:split));
    %TEST
    data_test.X = data.X([I(1), I(2)],ix(split:end));
    data_test.y = data.y(:,ix(split:end));

    for j=1:k
        clear model
        model = knnrule(data_train,j);
        %figure(j); ppatterns(data_test); pboundary(model);
        ypred = knnclass(data_test.X, model);
        err =[err cerror(ypred, data_test.y)*100];
        models = [models model];
    end
    %figure(i + 10);plot(err);
    err_matrix = [err_matrix; err];
    models_matrix = [models_matrix; models];
    models = [];
    err = [];
end

average = sum(err_matrix)/n_runs;
figure('Name', "Average of k-nn");
plot(average);

err_min = find(average==min(average));
disp(err_min + " - " +min(average));
clear model;
model_min = models_matrix(err_min(1));
figure('Name', "best model k-nn"); ppatterns(data_test); pboundary(model);

pred_y = knnclass(data_test.X, model_min);
test_y = data_test.y;
end

