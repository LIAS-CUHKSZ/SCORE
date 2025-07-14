%%%%
% Translation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% GT rotation vs FGO_PnL rotation estimates

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT
clear
clc
room_sizes =  [8,    6, 4;
               7,   7, 3;  
              10.5, 5, 3.5; 
              10.5, 6, 3.0];
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
scene_idx = 1;
dataset_name = dataset_ids(scene_idx);
space_size =  room_sizes(scene_idx,:);
data_folder="csv_dataset/"+dataset_name+"/";
load(data_folder+"lines3D.mat");
%%% rot params
prox_thres_r = 1*pi/180; % for clustering proximate stabbers
branch_reso_r = pi/256; % terminate bnb when branch size < branch_reso
sample_reso_r = pi/256; % resolution for interval analysis
epsilon_r = 0.015;
%%% trans params
branch_reso_t = 0.01; % terminate bnb when branch size <= branch_reso
prox_thres_t  = 0.01; %
epsilon_t = 0.03;
%%% statistics
column_names=["Image Id","# 2D lines","Outlier Ratio","IR Err Rot","IR Err Trans", "Rot Err","Trans Err", "time"];
columnTypes = ["int32"  ,"int32"     ,"double"       ,"double"    ,"double",       "double" ,"double"   ,"double"];
total_img=2000;
Record_est_SCM_entropy = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
parfor num =1:total_img
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_idx=num*10;     
    frame_id = sprintf("%06d",img_idx);
    if ~exist(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv",'file')     %%% read 2D line data of cur image
        continue
    end
    K_p=readmatrix(data_folder+"intrinsics/frame_"+frame_id+".csv");
    T_gt = readmatrix(data_folder+"poses/frame_"+frame_id+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/frame_"+frame_id+".csv")+1; % retrived sub-map
    retrived_closest_pose = readmatrix(data_folder+"retrived_closest_pose/frame_"+frame_id+".csv");
    [alpha,phi,theta] = rot2angle(retrived_closest_pose(1:3,1:3)');
    IR_err_rot = angular_distance(retrived_closest_pose(1:3,1:3),R_gt);
    IR_err_trans = norm(t_gt-retrived_closest_pose(1:3,4));
    %
    if abs(phi) < 3*pi/180    % ambiguous case
        west_east_flag = 2;
    else
        if phi<0
           west_east_flag = 1;
        else
           west_east_flag = 0;
        end
    end
    
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1]; % intrinsic matrix
    % lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1) 
    lines2D = readmatrix(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv"); 
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    % ---------------------------------------------------------------------
    % --- 2. skip image not observable ---
    with_match_idx = find(lines2D(:,9)>=0);
    M = length(with_match_idx);
    if M < 5
       fprintf("image "+num2str(img_idx)+" is ambigious in rotation, skip.\n");
       continue
    end
    % --- 3. semantic matching and saturation function design ---
    fprintf(num2str(img_idx)+"\n")
    lines3D_sub = lines3D(retrived_3D_line_idx,:);
    [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D_sub);  % match with clustered 3D lines
    num_2D_lines = size(lines2D,1);
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(ids==i);
    end
    kernel_buff_SCM_entropy = zeros(num_2D_lines,max(match_count));
    for i = 1:num_2D_lines
        kernel_buff_SCM_entropy(i,1)=1;
        for j = 2:match_count(i)
            kernel_buff_SCM_entropy(i,j)=log(j)-log(j-1);
        end
    end
    kernel_buff_SCM_entropy(:,2:end)=kernel_buff_SCM_entropy(:,2:end)/sum(sum(kernel_buff_SCM_entropy(:,2:end)));
    %
    outlier_ratio = 1-M/size(n_2D,1);
    %-------------------------------------------------------------
    %---- 4. complete pipeline starts here -----
    time_all = 0; 
    [R_opt_top,best_score,num_candidate_rot,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_SCM_entropy,...
        branch_reso_r,epsilon_r,sample_reso_r,prox_thres_r,west_east_flag);
    time_all = time_all+time;
    best_t_score = -1;
    best_R = eye(3);
    best_t = zeros(3,1);
    for n = 1:num_candidate_rot
        R_opt = R_opt_top(n*3-2:n*3,:)';
        [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
            under_specific_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r);
        %%% saturation function design
        match_count = zeros(num_2D_lines,1);
        for i = 1:num_2D_lines
            match_count(i) = sum(id_inliers_under_rot==i);
        end
        kernel_buff_SCM_entropy = zeros(num_2D_lines,max(match_count));
        for i = 1:num_2D_lines
            kernel_buff_SCM_entropy(i,1)=1;
            for j = 2:match_count(i)
                kernel_buff_SCM_entropy(i,j)=log(j)-log(j-1);
            end
        end
        kernel_buff_SCM_entropy(:,2:end)=kernel_buff_SCM_entropy(:,2:end)/sum(sum(kernel_buff_SCM_entropy(:,2:end)));
        %%% socre under t_gt
        residuals = sum(pert_rot_n_2D_inlier.*(endpoints_3D_inlier(1:2:end,:)-t_gt'),2);
        inliers_t_gt = find(abs(residuals )<=epsilon_t);
        score_t_gt_SCM_entropy = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_SCM_entropy);
        %%% Sat-CM: entropy + geo 
        [t_best_candidates,~,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_entropy,space_size,branch_reso_t,epsilon_t,prox_thres_t);
        time_all = time_all+time;
        % prune candidates according to geometric constraints
        [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM_entropy);
        t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
        if best_score > best_t_score
            best_t_score = best_score;
            best_R = R_opt;
            best_t = t_fine_tuned;
        end
    end
    rot_err = angular_distance(best_R,R_gt);
    t_err   = norm(best_t-t_gt);
    column_names=["Image Id","# 2D lines","Outlier Ratio","IR Err Rot","IR Err Trans", "Rot Err","Trans Err", "time"];
    Record_est_SCM_entropy(num+1,:)={img_idx,M,outlier_ratio,IR_err_rot,IR_err_trans,rot_err,t_err,time_all};
end
Record_est_SCM_entropy(Record_est_SCM_entropy.("Outlier Ratio")==0,:)=[];
%% 
num_valid = height(Record_est_SCM_entropy);
fprintf("============ time statistics ============\n")
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_est_SCM_entropy.time,[0.25,0.5,0.75]))
fprintf("============ rot err quantile ============\n")
fprintf("IR: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("IR Err Rot"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("Rot Err"),[0.25,0.5,0.75]))
fprintf("============ trans err quantile ============\n")
fprintf("IR: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("IR Err Trans"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("Trans Err"),[0.25,0.5,0.75]))
output_filename= "./matlab/Experiments/records/"+dataset_name+"_full_record.mat";
fprintf("============ Recall at 3/5/10 deg ============\n")
fprintf("IR: %f,%f,%f\n",sum(Record_est_SCM_entropy.("IR Err Rot")<[3,5,10])/num_valid*100)
fprintf("SCM_FGO_entropy: %f,%f,%f\n",sum(Record_est_SCM_entropy.("Rot Err")<[3,5,10])/num_valid*100)
fprintf("============ Recall at 5cm/10cm/20cm ============\n")
fprintf("IR: %f,%f,%f\n",sum(Record_est_SCM_entropy.("IR Err Trans")<[0.05,0.1,0.2])/num_valid*100)
fprintf("SCM_FGO_entropy: %f,%f,%f\n",sum(Record_est_SCM_entropy.("Trans Err")<[0.05,0.1,0.2])/num_valid*100)
save(output_filename);
%%
% ---------------------------------------------------------------------
% --- sub-functions ---
function flag = checkTransAmbiguity(img_idx,lines2D,R_gt)
    M = size(lines2D,1);
    A_gt = zeros(M,3);
    flag = false;
    for i=1:M
        n = lines2D(i,1:3); 
        A_gt(i,:)=(R_gt*n')';
    end
    if rank(A_gt'*A_gt)<3
        flag = true;
        fprintf("image"+num2str(img_idx)+" is ambigious in translation, skip.\n");
    end
end


function [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
    under_specific_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r)

    inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
    id_inliers_under_rot = ids(inlier_under_rot);
    n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
    endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
    %%% fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
    pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
end

function [best_score,t_real_candidate] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D,endpoints_3D,ids,epsilon_t,t_candidates,kernel_buff)
    p_3D = endpoints_3D(1:2:end,:);
    best_score = -1; t_real_candidate=[];
    for k=1:size(t_candidates,2)
        t_test = t_candidates(:,k);
        residuals = sum(pert_rot_n_2D.*(p_3D-t_test'),2);
        inliers = find(abs(residuals )<=epsilon_t);
        %%% the above inliers satisfy the geometric constraints,
        %%% we urther filter lines behind the camera and outside image
        real_inliers = prune_inliers(R_opt,intrinsic,inliers,endpoints_3D,t_test);
        score=calculate_score(ids(real_inliers),kernel_buff);
        if score > best_score
            best_score = score; t_real_candidate = t_test;
        elseif score == best_score
            t_real_candidate = [t_real_candidate,t_test];
        else
        end
    end
end
