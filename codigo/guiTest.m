function [Accuracy, Accuracy_std , Sensitivity, Sensitivity_std, Specificity, Specificity_std, FScore, FScore_std] = guiTest(scenario, features, n_features, model_choice, n_iterations, cost)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

scenario = str2num(scenario);
n_iterations = str2num(n_iterations);

acc = [];
spe = [];
sen = [];
fsc = [];
for i= 1:n_iterations

    %Scenario
    data=load('data.mat').data;
    data = chooseScenario(data, scenario);
    
    %Feature Selection
    n_features = str2num(n_features);
    
    
    [data_train, data_test] = splitDataset(data, 200000);
    switch features
        case 'PCA'
    
            data_train=scalestd(data_train);
            data_test=scalestd(data_test);
            
            model = pca(data_train.X,n_features);
            data_train.X=model.W'*data_train.X+model.b;
            data_test.X=model.W'*data_test.X+model.b;
    
        case 'Kruskal-Wallis'
            data_train.X=data_train.X([4,7,9,11,13],:);
            data_test.X=data_test.X([4,7,9,11,13],:);
    
        case 'Fishers LDA'
            [data_train,data_test]=ldaFisher(data_train,data_test);
        case 'ROC Curve'
            data_train.X=data_train.X([4,7,9,11,13],:);
            data_test.X=data_test.X([4,7,9,11,13],:);
        otherwise
            warning('Unexpected Value');
    end
    
    
    %Classification
    switch model_choice
        case 'Linear MDC'
            [pred_y,true_y] = mdClassifier(data_train,data_test);
        case 'Mahal MDC'
            [pred_y,true_y] = mahalClassifier(data_train,data_test);
        case 'Fishers LD'
            [pred_y,true_y] = mdClassifier(data_train,data_test);
        case 'Bayes Classifier'
            [pred_y,true_y]=bayesClassifier(data_train,data_test,cost);
        case 'KNN Classifier'
            [pred_y,true_y] = knn(data, n_runs, k);
        case 'SVMs'
            constraint=1;
            gamma=1;
            func='rbf';
            [pred_y,true_y]=svmClassifier(data_train,data_test,func,constraint,gamma);
       otherwise
            warning('Unexpected Value');
    end
    
    %Compute Performance
    
    [Accuracy, Specificity, Sensitivity, FScore] = computePerformance(pred_y,true_y);
    disp([Accuracy, Specificity, Sensitivity, FScore]);
    C = confusionmat(true_y,pred_y);
    confusionchart(C,[0 1]);
    acc = [acc Accuracy];
    spe = [spe Specificity];
    sen = [sen Sensitivity];
    fsc = [fsc FScore];
end

Accuracy = mean(acc);
Accuracy_std = std(acc);

Specificity = mean(spe);
Specificity_std = mean(spe);

Sensitivity = mean(sen);
Sensitivity_std = std(sen);

FScore = mean(fsc);
FScore_std = std(fsc);
end

