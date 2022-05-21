function [Accuracy, Accuracy_std , Sensivity, Sensivity_std, Specificity, Specificity_std, FScore, FScore_std] = guiTest(scenario, features, n_features, model, n_iterations)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

scenario = str2num(scenario);
n_iterations = str2num(n_iterations);


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
        model = pca(data_train.X);

    case 'Kruskal-Wallis'
        for i=1:data_train.dim
            [p,atab,stats]=kruskalwallis(data_train.X(i,:),data_train.y,'off');
            rank{i,1}=data_train.indep_names{i};
            rank{i,2}=atab{2,5};
            atab;
        end

    case 'Fishers LDA'
        [data_train,data_test]=ldaFisher(data_train,data_test);

    case 'ROC Curve'
        %TODO
    otherwise
        warning('Unexpected Value');
end


%Classification

switch model
    case 'Linear MDC'
        [pred_y,true_y]=mdClassifier(data_train,data_test);
    case 'Mahal MDC'
        %TODO
    case 'Fishers LD'
        %TODO
    case 'Bayes Classifier'
        [pred_y,true_y]=bayesClassifier(data_train,data_test,costs);
    case 'KNN Classifier'
        %MOST LIKELY MAL
        %TODO
        [pred_y,true_y]=knn(data_train, n_runs, k);
    case 'SVMs'
        [pred_y,true_y]=svmClassifier(data_train,data_test,func,constraint,gamma)
   otherwise
        warning('Unexpected Value');
end

%Compute Performance

[Accuracy, Specificity,Sensitivity, FScore] = computePerformance(pred_y,true_y);
Accuracy_std = 0;
Sensitivity_std = 0;
Specificity_std = 0;
FScore_std = 0;
end

