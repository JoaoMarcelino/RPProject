function [accuracy,specificity,sensitivity, fscore] = computePerformance(pred_y,true_y)
    
    accuracy=sum(pred_y==true_y)/size(true_y,2);
    tp=sum(pred_y==1 & true_y==1);
    fn=sum(pred_y==2 & true_y==1);
    tn=sum(pred_y==2 & true_y==2);
    fp=sum(pred_y==1 & true_y==2);

    specificity=tn/(tn+fp);
    sensitivity=tp/(tp+fn);
    fscore = tp/(tp + 1/2 * (fp + fn));
end

