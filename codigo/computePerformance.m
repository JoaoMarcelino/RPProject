function [accuracy,specificity,sensibility] = computePerformance(pred_y,true_y)
    accuracy=sum(pred_y==true_y)/size(true_y,2);
    tp=sum(pred_y==0 & true_y==0);
    fn=sum(pred_y==1 & true_y==0);
    tn=sum(pred_y==1 & true_y==1);
    fp=sum(pred_y==0 & true_y==1);

    specificity=tn/(tn+fp);
    sensibility=tp/(tp+fn);
end
