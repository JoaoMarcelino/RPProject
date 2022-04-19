function [pred_y,true_y] = mahalClassifier(data_train,data_test)
    classes=sort(unique(data_train.y));
    centroids=[];

    for i=1:length(classes)
        X=data_train.X(:,find(data_train.y==classes(i)));
        centroid=sum(X,2)./size(X,2);
        centroids=[centroids, centroid];
    end

    C=0;
    for k=1:size(centroids,2)
            c=cov(data_test.X(data_test.y==classes(k)));
            C=C+c;
    end
    C=C./size(classes,2);
    invC=inv(C);
    pred_y=[];
    for i=1:size(data_test.X,2)
        gxVector=[];
        for k=1:size(centroids,2)
            gx=centroids(:,k)'*invC*data_test.X(:,i)-0.5*centroids(:,k)'*invC*centroids(:,k);
            gxVector=[gxVector,gx];
        end
        [~,index]=max(gxVector);

        pred_y=[pred_y,classes(index)];    
    end
    true_y=data_test.y;
end

