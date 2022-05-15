function [areaUnderCurve] = rocArea(FPRValues,FNRValues)
%ROCAREA Summary of this function goes here
%   Detailed explanation goes here
SS=1-FNR;
SP=1-FPR;
AUC=0;
for j=2:numel(FPR)
    AUC=AUC+(SS(j)*(FPR(j-1)-FPR(j)));
end
areaUnderCurve=AUC;
end

