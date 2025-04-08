%%%%
% Implementation of the FGO translation ransac estimator under saturated consensus maximization

%%% Inputs:
% vector_n      : N x 3, the normal vector paramater for each 2D image line.
% point_c       : N x 3, a point on the 3D line.
% R_opt         : optimal orientation from upstream estimation.
% id_inliers    : N x 1, id of the 2D lines.
% sampleSize    : parameter for ransac
% max_iter      : parameter for ransac
% kernel        : saturation function
% room_size     : 3 x 1

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [t_opt,num_candidate,max_score] = Sat_t_ransac(vector_n,point_c,R_opt,id_inliers,sampleSize,max_iter,epsilon_t,kernel,room_size)
%%% group matched line pairs of a same 2d line
id_set = unique(id_inliers);
num_index = length(id_set);
match_group = cell(num_index,1); % record each 2d line's matchings
for i=1:num_index
    temp_index = find(id_inliers==id_set(i));
    if ~isempty(temp_index)
        match_group{i}=temp_index;
    end
end
%%%
N = length(vector_n);
A_all=zeros(3,N);
for i=1:N
    ni=vector_n(:,i);
    A_all(:,i) = R_opt*ni; 
end
b_all = dot(A_all,point_c)';
%%%
iter=1;
opt_inliers=[];
max_score=0;
t_opt = 100*ones(3,1);
while(iter<max_iter)
    A=zeros(sampleSize,3);
    b=zeros(sampleSize,1);
    random_num = randperm(num_index,sampleSize); % randomly pick 2D lines
    for j=1:sampleSize % randomly pick a matched 3D line for each picked 2D line 
        group = match_group{random_num(j)};
        choosen_idx = group(randperm(length(group),1));
        A(j,:) = A_all(:,choosen_idx)';
        b(j) = b_all(choosen_idx);
    end
    t_sample = pinv(A'*A)*(A'*b); 
    inliers = abs(A_all'*t_sample-b_all)<=epsilon_t; 
    if sum(inliers)<sampleSize
        iter = iter+1;
        continue
    end
    A = A_all(:,inliers)';
    b = b_all(inliers);
    t_sample = pinv(A'*A)*(A'*b); 
    if t_sample(1)<=0 || t_sample(1)>=room_size(1) || t_sample(2)<=0 || t_sample(2)>=room_size(2) || t_sample(3)<=0 || t_sample(3)>=room_size(3)
        continue
    end
    inliers = abs(A_all'*t_sample-b_all)<=epsilon_t; 
    score=calculate_score(id_inliers(inliers),kernel);
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

