function [data_train,data_test] = splitDataset(data,trainSize)
    ix = randperm(data.num_data);
    trainSize=floor(trainSize);

    x_training = ix(1:trainSize);
    x_testing = ix(trainSize+1:end);
    
    %Training set
    data_train.X = data.X(:,x_training);
    data_train.y = data.y(x_training);
    data_train.dim = size(data_train.X,1);
    data_train.num_data = size(data_train.X,2);
    data_train.name = "Training Dataset";
    data_train.indep_names = data.indep_names;
    data_train.dep_names = data.dep_names;
   
    %Testing set
    data_test.X = data.X(:,x_testing);
    data_test.y = data.y(x_testing);
    data_test.dim = size(data_test.X,1);
    data_test.num_data = size(data_test.X,2);
    data_test.name = "Testing Dataset";
    data_test.indep_names = data.indep_names;
    data_test.dep_names = data.dep_names;

end

