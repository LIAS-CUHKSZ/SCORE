%%%%
% Semantic Matching

%%% Inputs:
% lines2D:          K  x 4, normal vector(3x1), semantic label(1)
% lines3D:          M  x 7 , endpoint a(3x1), endpoint b(3x1), semantic label(1) 

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [ids_2D,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D)
    % pre-allocation space
    total_match_num=0; 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1); % find 3D lines with same semantic label
        total_match_num = total_match_num+length(idx_matched_3D);
    end
    v_3D=zeros(total_match_num,3);
    n_2D=zeros(total_match_num,3);
    endpoints_3D = zeros(total_match_num*2,3);
    ids_2D=zeros(total_match_num,1); % record id of the 2D line
    temp=0;
    % fill in 
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D(:,7)-lines2D(i,4))<0.1);
        num_matched=length(idx_matched_3D);
        for j = 1:num_matched
            ids_2D(temp+j)=i;
            n_2D(temp+j,:) = lines2D(i,1:3); % normal vector of the 2D line
            v = lines3D(idx_matched_3D(j),4:6)-lines3D(idx_matched_3D(j),1:3);
            v_3D(temp+j,:) = v/norm(v); % direction vector of the 3D line
            endpoints_3D(2*(temp+j)-1,:) = lines3D(idx_matched_3D(j),1:3); % endpoints of the 3D line
            endpoints_3D(2*(temp+j),:) = lines3D(idx_matched_3D(j),4:6);
        end
        temp=temp+num_matched;
    end
end

