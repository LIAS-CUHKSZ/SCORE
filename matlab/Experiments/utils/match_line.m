function [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D)
    % pre-allocation space
    total_match_num=0; 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1);
        total_match_num = total_match_num+length(idx_matched_3D);
    end
    v_3D=zeros(total_match_num,3);
    n_2D=zeros(total_match_num,3);
    endpoints_3D = zeros(total_match_num*2,3);
    ids=zeros(total_match_num,1); % record id of the 2D line
    temp=0;
    % fill in 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1);
        num_matched=length(idx_matched_3D);
        for j = 1:num_matched
            ids(temp+j)=i;
            n_2D(temp+j,:) = lines2D(i,1:3);
            v = lines3D(idx_matched_3D(j),4:6)-lines3D(idx_matched_3D(j),1:3);
            v_3D(temp+j,:) = v/norm(v);
            endpoints_3D(2*(temp+j)-1,:) = lines3D(idx_matched_3D(j),1:3);
            endpoints_3D(2*(temp+j),:) = lines3D(idx_matched_3D(j),4:6);
        end
        temp=temp+num_matched;
    end
end

