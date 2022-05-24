% Kruskal + KNN
close all;
clear all;
C=[-5:6];
Gamma=[-5:2];
C_pot=2.^C;
Gamma_pot=2.^Gamma;
values=zeros([length(C_pot),length(Gamma_pot)]);
x=1;
for g=Gamma_pot
    y=1;
    for c=C_pot
        data=load('data.mat').data;
        [data_train,data_test]=splitDataset(data,200000);
        data_train=chooseScenario(data_train,1);
        data_test=chooseScenario(data_test,1);
        data_train=resample(data_train,1000);

        %kruskall
        data_train.X=data_train.X([4,7,9,11,13],:);
        data_test.X=data_test.X([4,7,9,11,13],:);
        
        model = fitcsvm(data_train.X',data_train.y','KernelFunction','rbf','BoxConstraint',c,'kernelScale',g,'Standardize',true);
        [pred_y]=predict(model,data_test.X')';
        true_y=data_test.y;
        [accuracy,specificity,sensibility,fscore]=computePerformance(pred_y,true_y);
        values(x,y)=fscore;
        y=y+1
    end
    x=x+1;
end
figure;
[X,Y] = meshgrid(Gamma_pot,C_pot);
contourf(X,Y,values');
