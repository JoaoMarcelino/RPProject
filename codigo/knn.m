function [pred_y,true_y] = knn(data_train, data_test, k)
    model = fitcknn(data_train.X',data_train.y','NumNeighbors',k,'Standardize',1);
    pred_y= predict(model,data_test.X')';
    true_y=data_test.y;
end

