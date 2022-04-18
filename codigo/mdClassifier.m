function [pred_y,true_y] = mdClassifier(data_train,data_test)
    classes=sort(unique(data_train.y));
    centroids=[];

    for i=1:length(classes)
        X=data_train.X(:,find(data_train.y==classes(i)));
        centroid=sum(X,2)./size(X,2);
        centroids=[centroids, centroid];
    end

    pred_y=[];
    for i=1:size(data_test.X,2)
        closest=Inf;
        rightCluster=-1;
        for k=1:size(centroids,2)
            distance=centroids(:,k)-data_test.X(:,i);
            distance=centroids(:,k)-data_test.X(:,i);
            distance=sum(distance.*distance);
            if distance<closest
                closest=distance;
                rightCluster=classes(k);
            end
        end
        pred_y=[pred_y,rightCluster];
        
    end
    true_y=data_test.y;
end

