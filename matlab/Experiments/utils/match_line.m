function [n_2D,v_3D,id,p_3D]=match_line(lines2D,lines3D)
    % pre-allocation space
    total_match_num=0; 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1);
        total_match_num = total_match_num+length(idx_matched_3D);
    end
    v_3D=zeros(total_match_num,3);
    p_3D=zeros(total_match_num,3);
    n_2D=zeros(total_match_num,3);
    id=zeros(total_match_num,1); % record id of the 2D line
    temp=0;
    % fill in 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1);
        num_matched=length(idx_matched_3D);
        for j = 1:num_matched
            n_2D(temp+j,:) = lines2D(i,1:3);
            v_3D(temp+j,:) = lines3D(idx_matched_3D(j),4:6);
            p_3D(temp+j,:) = lines3D(idx_matched_3D(j),1:3);
            id(temp+j)=i;
        end
        temp=temp+num_matched;
    end
end

