function [None] = svm(data)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes herec


error_matrix= [];
for n_runs=1:10
    
    ix = randperm(data.num_data);
    trainSize=floor(size(data.X,2)/2);
    
    x_training = ix(1:trainSize);
    x_testing = ix(trainSize+1:end);
        
    %Training set
    data_train.X = data.X(:,x_training);
    data_train.y = data.y(:,x_training);
    data_train.dim = size(data_train.X,1);
    data_train.num_data = size(data_train.X,2);
    data_train.name = "Training Dataset";
    data_train.indep_names = data.indep_names;
    
    %Testing set
    data_test.X = data.X(:,x_testing);
    data_test.y = data.y(:,x_testing);
    data_test.dim = size(data_test.X,1);
    data_test.num_data = size(data_test.X,2);
    data_test.name = "Testing Dataset";
    data_test.indep_names = data.indep_names;
    
    
    errors= [];
    for c= 2^-5:2:2^12
        halfsies = [];
        for gamma = 2^-30:2:2^5
           model = fitcsvm(data_train.X',data_train.y','KernelFunction','rbf','BoxConstraint', c, 'KernelScale', gamma);
           %figure; ppatterns(data_train); 
           [ypred] =predict(model, data_train.X'); 
           err = cerror( ypred', data_test.y );
           halfsies= [halfsies err];
        end
        errors = [errors mean(halfsies)];
    end

    error_matrix = [error_matrix ; errors];
end

av = mean(error_matrix);

figure;plot(av);
end

