function [data_train,data_test] = ldaFisher(data_train,data_test)

    X1=data_train.X(:,find(data_train.y==0));
    X2=data_train.X(:,find(data_train.y==1));
    m1=sum(X1,2)./size(X1,2);
    m2=sum(X2,2)./size(X2,2);
    S1=0;
    S2=0;
    for i=1:size(X1,2)
        S1=S1+(X1(:,i)-m1)*(X1(:,i)-m1)';
    end
    for i=1:size(X2,2)
        S2=S2+(X2(:,i)-m2)*(X2(:,i)-m2)';
    end
    Sw=S1+S2;
    w=inv(Sw)*(m1-m2);
    
    data_train.X=w'*data_train.X;
    data_test.X=w'*data_test.X;
   
end

