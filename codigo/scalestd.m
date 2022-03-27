function [dataNormalized] = scalestd(data)
    meanData=mean(data.X,2);
    standardDevData=std(data.X,[],2);

    for i=1:size(data.X,1)
        data.X(i,:)=(data.X(i,:)-meanData(i))/standardDevData(i);
        
    end
    dataNormalized=data;
end

