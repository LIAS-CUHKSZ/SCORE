function [t_opt,num_candidate,max_score] = sat_var_t_Ransac(data_2d_N,data_3d_c,R_opt,id_inliers,sampleSize,ransac_iterations,epsilon_t,kernel,room_size)
%%%% group matching under a same 2d line
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
num_mathcing = length(data_2d_N);
A_all=zeros(3,num_mathcing);
for i=1:num_mathcing
    ni=data_2d_N(:,i);
    A_all(:,i) = R_opt*ni; 
end
b_all = dot(A_all,data_3d_c)';
%%%%
iter=1;
opt_inliers=[];
max_score=0;
t_opt = 100*ones(3,1);
while(iter<ransac_iterations)
    A=zeros(sampleSize,3);
    b=zeros(sampleSize,1);
    random_num = randperm(num_index,4);
    for j=1:sampleSize
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

