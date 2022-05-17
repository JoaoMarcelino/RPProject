function [data] = chooseScenario(data, nScenario)
    
    %scenario A - CoronaryHeartDisease: 1-Positive/2-Negative
    if nScenario == 1
        data.y=data.y(1,:);
    end
    %scenario B - HeartDisease: 1-Positive (CoronaryHeartDisease or MyocardialInfarction)/2-Negative
    if nScenario == 2
        index = find(data.y(1,:) == 1  | data.y(2,:) == 1);
        data.y = ones(1, data.num_data).*2;
        data.y(index) = 1;
    end
    
    %scenario B - Heart Disease with comorbidities
    if nScenario == 3
        index_noHeartDiseases = find(data.y(1,:) == 2  & data.y(2,:) == 2);
        index_HeartDiseases_noComorbidities = find((data.y(1,:) == 1  | data.y(2,:) == 1) & data.y(3,:) == 2 & data.y(4,:) == 2);
        data.y = ones(1, data.num_data);
        data.y(index_HeartDiseases_noComorbidities) = 2;
        data.y(index_noHeartDiseases) = 3;
    end

end
