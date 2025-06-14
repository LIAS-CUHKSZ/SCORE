function score=calculate_score(inlier_ids,kernel_buffer)
    trunc_num = length(kernel_buffer);
    score=0;
    unique_ids=unique(inlier_ids);
    for i=1:length(unique_ids)
        num = sum(inlier_ids==unique_ids(i));
        num = min(num,trunc_num);
        for j=1:num
            score=score+kernel_buffer(j);
        end
    end
end