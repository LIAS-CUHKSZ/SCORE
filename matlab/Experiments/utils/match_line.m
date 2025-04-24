function [data_2D_n,data_3D_v,id]=match_line(lines2D,lines3D)
    % pre-allocation space
    total_match_num=0; 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1);
        total_match_num = total_match_num+length(idx_matched_3D);
    end
    data_3D_v=zeros(total_match_num,3);
    data_2D_n=zeros(total_match_num,3);
    id=zeros(total_match_num,1); % record id of the 2D line
    temp=0;
    % fill in 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1);
        num_matched=length(idx_matched_3D);
        for j = 1:num_matched
            data_2D_n(temp+j,:) = lines2D(i,1:3);
            data_3D_v(temp+j,:) = lines3D(idx_matched_3D(j),4:6);
            id(temp+j)=i;
        end
        temp=temp+num_matched;
    end
end

