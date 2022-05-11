function [data_train,data_test] = chooseScenario(nSplit, nScenario)

data=load('data.mat').data;
[data_train,data_test]=splitDataset(data,nSplit);


%scenario A - CoronaryHeartDisease: 0/1
if nScenario == 1
    data_train.y=data_train.y(1,:)-1;
    data_test.y=data_test.y(1,:)-1;
end
%scenario B - HeartDisease: 0/1
if nScenario == 2
    data_train.y=data_train.y(1:2,:)-1;
    index = find(data_train.y(1,:) == 1  & data_train.y(2,:) == 1);
    data_train.y = zeros(1, data_train.num_data);
    data_train.y(index) = 1;

    data_test.y=data_test.y(1:2,:)-1;
    index = find(data_test.y(1,:) == 1  & data_test.y(2,:) == 1);
    data_test.y = zeros(1, data_test.num_data);
    data_test.y(index) = 1;
end
if nScenario == 3
    data_train.y=data_train.y(1:4,:)-1;
    index_noHeartDiseases = find(data_train.y(1,:) == 1  & data_train.y(2,:) == 1);
    index_HeartDiseases_noComorbidities = find((data_train.y(1,:) == 0  | data_train.y(2,:) == 0) & (data_train.y(3,:) == 1 | data_train.y(4,:) == 1));
    data_train.y = zeros(1, data_train.num_data);
    data_train.y(index_noHeartDiseases) = 2;
    data_train.y(index_HeartDiseases_noComorbidities) = 1;

    
    data_test.y=data_test.y(1:4,:)-1;
    index_noHD = find(data_test.y(1,:) == 1  & data_test.y(2,:) == 1);
    index_HD_noC = find((data_test.y(1,:) == 0  | data_test.y(2,:) == 0) & (data_test.y(3,:) == 1 | data_test.y(4,:) == 1));
    data_test.y = zeros(1, data_test.num_data);
    data_test.y(index_noHD) = 2;
    data_test.y(index_HD_noC) = 1;
end
end
