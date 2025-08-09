% calculate score based on inlier ids and sat function.
function score=calculate_score(inlier_ids,sat_buffer)
    score=0;
    unique_ids=unique(inlier_ids);
    for i=1:length(unique_ids)
        id = unique_ids(i);
        num = sum(inlier_ids==id);
        for j = 1:num
            score = score + sat_buffer(i,j);
        end
    end
end