function score=calculate_score(inlier_ids,kernel)
    score=0;
    unique_ids=unique(inlier_ids);
    for i=1:length(unique_ids)
        num = sum(inlier_ids==unique_ids(i));
        for j=1:num
            score=score+kernel(j);
        end
    end
end

