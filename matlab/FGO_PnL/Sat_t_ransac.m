function [t_opt,num_candidate,best_score] = sat_t_Ransac(n_2D,endpoints_3D,R_opt,id_inliers,sampleSize,ransac_iterations,epsilon_t,room_size)
kernel =  @(x) x^(-9);
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
num_mathcing = length(n_2D);
A_all=zeros(3,num_mathcing);
for i=1:num_mathcing
    ni=n_2D(:,i);
    A_all(:,i) = R_opt*ni;
end
b_all = dot(A_all,p_3D)';
%%%%
iter=1; best_score=0;
t_opt = 100*ones(3,1); opt_inliers=[];
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
    if rank(A'*A)<3
        continue
    end
    t_sample = (A'*A)\(A'*b);
    inliers = abs(A_all'*t_sample-b_all)<=epsilon_t;
    A = A_all(:,inliers)'; b = b_all(inliers);
    if rank(A'*A)<3
        continue
    end
    t_sample = pinv(A'*A)*(A'*b);
    if check_outof_room(t_sample,room_size)
        iter = iter +1;
        continue
    end
    inliers = find(abs(A_all'*t_sample-b_all)<=epsilon_t);
    %%% the above inliers satisfy the geometric constraints,
    %%% we urther filter lines behind the camera
    delete = [];
    for k=1:length(inliers)
        end_point_1 = R_opt'*(endpoints_3D(inliers(k)*2-1,:)'-t_sample);
        end_point_2 = R_opt'*(endpoints_3D(inliers(k)*2,:)'-t_sample);
        if end_point_1(3) < 0 && end_point_2(3)<0 %% filter out the lines behind the camera
            delete=[delete,k];
            continue
        end
    end
    if ~isempty(delete)
        inliers(delete)=[];
    end
    score=calculate_score(id_inliers(inliers),kernel);
    if score>best_score
        best_score=score; opt_inliers=inliers; t_opt = t_sample;
    elseif score==best_score && min(vecnorm(t_sample-t_opt))>0.03
        opt_inliers=[opt_inliers,inliers];
        t_opt = [t_opt,t_sample];
    end
    iter=iter+1;
end
num_candidate = size(opt_inliers,2);
end


function flag = check_outof_room(t_test,room_size)
flag =   t_test(1)<=0 || t_test(1)>=room_size(1) || t_test(2)<=0 || t_test(2)>=room_size(2) || t_test(3)<=0 || t_test(3)>=room_size(3);
end

%%% deprecated
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