%%%%
% Implementation of the FGO translation ransac estimator under saturated consensus maximization

%%% Inputs:
% vector_n      : N x 3, the normal vector paramater for each 2D image line.
% point_c       : N x 3, a point on the 3D line.
% R_opt         : optimal orientation from upstream estimation.
% id            : N x 1, id of the 2D lines.
% intrinsic     : 3 x 3, camera intrinsic matrix
% endpoints_3d  : 2N x 3, end points of the 3D line.
% sampleSize    : parameter for ransac
% max_iter      : parameter for ransac
% kernel        : saturation function
% room_size     : 3 x 1, boundary of the room

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [t_opt,num_candidate,max_score] = Sat_t_ransac(vector_n,point_c,R_opt,endpoints_3d,id,sampleSize,max_iter,epsilon_t,kernel,room_size,intrinsic)
%%% preprocessing 
% group matching under a same 2d line
id_set = unique(id);
num_index = length(id_set);
match_group = cell(num_index,1); % record each 2d line's matchings
for i=1:num_index
    temp_index = find(id==id_set(i));
    if ~isempty(temp_index)
        match_group{i}=temp_index;
    end
end
%
num_mathcing = length(vector_n);
A_all=zeros(3,num_mathcing);
for i=1:num_mathcing
    ni=vector_n(:,i);
    A_all(:,i) = R_opt*ni; 
end
b_all = dot(A_all,point_c)';

%%% RANSAC
% initialization
iter=1;
opt_inliers=[];
max_score=0;
t_opt = 100*ones(3,1);
% start iteration
while(iter<max_iter)
    A=zeros(sampleSize,3);
    b=zeros(sampleSize,1);
    random_num = randperm(num_index);% randomly sample a 2D line
    for j=1:sampleSize  % randomly sample a matched 3D line for each randomly sampled 2D line
        group = match_group{random_num(j)};
        choosen_idx = group(randperm(length(group),1));
        A(j,:) = A_all(:,choosen_idx)';
        b(j) = b_all(choosen_idx);
    end
    t_sample = pinv(A'*A)*(A'*b); 
    inliers = find(abs(A_all'*t_sample-b_all)<=epsilon_t); 
    if length(inliers)<sampleSize
        iter = iter+1;
        continue
    end
    A = A_all(:,inliers)';
    b = b_all(inliers);
    t_sample = pinv(A'*A)*(A'*b);  % update the estimate
    if sum(t_sample>room_size')+sum(t_sample<0)~=0 % if the position is out of the room
        continue
    end
    inliers = find(abs(A_all'*t_sample-b_all)<=epsilon_t);
    %%% filter inliers
    delete = []; % index of inliers to be filtered 
    % filter out the lines behind the camera
    for k=1:length(inliers)
        end_point_1 = R_opt'*(endpoints_3d(:,inliers(k)*2-1)-t_sample);
        end_point_2 = R_opt'*(endpoints_3d(:,inliers(k)*2)-t_sample);
        if end_point_1(3) < 0 && end_point_2(3)<0 
            delete=[delete,k];
            continue
        end
    end
    if ~isempty(delete)
        inliers(delete)=[];
    end
    % update optimal estimate
    score=calculate_score(id(inliers),kernel);
    if score>max_score
        opt_inliers=inliers;
        max_score=score;
        t_opt = t_sample;
    elseif score==max_score && min(vecnorm(t_sample-t_opt))>epsilon_t
        opt_inliers=[opt_inliers,inliers];
        t_opt = [t_opt,t_sample];
    end
    iter=iter+1;
end
num_candidate = size(opt_inliers,2);
end

%%% deprecated code ↓
% if end_point_1(3) < 0
%     dif = end_point_2-end_point_1;
%     end_point_1 = end_point_1-dif*(end_point_1(3)-0.1)/dif(3);
% end
% if end_point_2(3) < 0
%     dif = end_point_1-end_point_2;
%     end_point_2 = end_point_2-dif*(end_point_2(3)-0.1)/dif(3);     
% end
% end_point_1_pixel = intrinsic*end_point_1;
% end_point_1_pixel = end_point_1_pixel(1:2)/end_point_1_pixel(3);
% end_point_2_pixel = intrinsic*end_point_2;
% end_point_2_pixel = end_point_2_pixel(1:2)/end_point_2_pixel(3);
% intersect_flag = checkLineRect(end_point_1_pixel,end_point_2_pixel,1920,1440);
% if ~intersect_flag
%    delete=[delete,k];
% end