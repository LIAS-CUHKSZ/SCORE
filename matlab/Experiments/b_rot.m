%%%%
% Rotation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% FGO_PnL vs EGO_PnL

%%% Author:  Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT

clear;
clc;
dataset_ids = ["689fec23d7","c173f62b15","69e5939669","a1d9da703c"];
dataset_idx = dataset_ids(4);
data_folder="csv_dataset/"+dataset_idx+"/";
load(data_folder+"lines3D.mat");
%%% statistics
total_img=1000;
column_names=...
    ["Image ID","# 2D lines","epsilon_r","Outlier Ratio","IR Err Rot","Max Rot Err","Min Rot Err","# Rot Candidates","Best Score","GT Score","Time","Rot Vec"];
columnTypes =...
    ["int32","int32","double","double","double","double","double","int32","double","double","double","cell"];
Record_CM_FGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO_exp       =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO_power     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO_entropy   =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);

%%%  params
prox_thres = 1*pi/180; % for clustering proximate stabbers
branch_reso = pi/256; % terminate bnb when branch size < branch_reso
sample_reso = pi/256; % resolution for interval analysis
epsilon_r = 0.015;
%%
parfor num = 0:2000
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
    retrived_err_rot = angular_distance(retrived_closest_pose(1:3,1:3),R_gt);
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
    % epsilon_r = max(lines2D(with_match_idx,10))+0.001;
    % --- 3. semantic matching and saturation function design ---
    fprintf(num2str(img_idx)+"\n")
    lines3D_sub = lines3D(retrived_3D_line_idx,:);
    [ids,n_2D,v_3D,~]=match_line(lines2D,lines3D_sub);  % match with clustered 3D lines
    num_2D_lines = size(lines2D,1);
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(ids==i);
    end
    kernel_buff_SCM_power   = zeros(num_2D_lines,max(match_count));
    kernel_buff_SCM_exp     = zeros(num_2D_lines,max(match_count));
    kernel_buff_SCM_entropy = zeros(num_2D_lines,max(match_count));
    for i = 1:num_2D_lines
        kernel_buff_SCM_power(i,1)=1;
        kernel_buff_SCM_exp(i,1)=1;
        kernel_buff_SCM_entropy(i,1)=1;
        for j = 2:match_count(i)
            kernel_buff_SCM_power(i,j) = j^(-8);
            kernel_buff_SCM_exp(i,j)=2^(-j);
            kernel_buff_SCM_entropy(i,j)=log(j)-log(j-1);
        end
    end
    kernel_buff_SCM_power(:,2:end)=kernel_buff_SCM_power(:,2:end)/sum(sum(kernel_buff_SCM_power(:,2:end)));
    kernel_buff_SCM_exp(:,2:end)=kernel_buff_SCM_exp(:,2:end)/sum(sum(kernel_buff_SCM_exp(:,2:end)));
    kernel_buff_SCM_entropy(:,2:end)=kernel_buff_SCM_entropy(:,2:end)/sum(sum(kernel_buff_SCM_entropy(:,2:end)));
    kernel_buff_CM=ones(num_2D_lines,max(match_count));
    gt_inliers_idx = find(abs(dot(R_gt'*v_3D',n_2D'))<=epsilon_r);
    gt_inliers_id = ids(gt_inliers_idx);
    % ---------------------------------------------------------------------
    % --- 4. estimate rotation with Consensus Maximization  ---
    gt_score = calculate_score(gt_inliers_id,kernel_buff_CM);
    % CM_FGO
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_CM,...
        branch_reso,epsilon_r,sample_reso,prox_thres,west_east_flag);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_CM_FGO(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};

    % ---------------------------------------------------------------------
    % --- 5. estimate rotation with Saturated Consensus Maximization  ---
    % SCM_FGO_power
    gt_score = calculate_score(gt_inliers_id,kernel_buff_SCM_power);
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_SCM_power,...
        branch_reso,epsilon_r,sample_reso,prox_thres,west_east_flag);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_SCM_FGO_power(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};

    % SCM_FGO_exp
    gt_score = calculate_score(gt_inliers_id,kernel_buff_SCM_exp);
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_SCM_exp,...
        branch_reso,epsilon_r,sample_reso,prox_thres,west_east_flag);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_SCM_FGO_exp(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};

    % SCM_FGO_entropy
    gt_score = calculate_score(gt_inliers_id,kernel_buff_SCM_entropy);
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_SCM_entropy,...
        branch_reso,epsilon_r,sample_reso,prox_thres,west_east_flag);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_SCM_FGO_entropy(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};
   
end
Record_CM_FGO(Record_CM_FGO.("Best Score")==0,:)=[];
Record_SCM_FGO_power(Record_SCM_FGO_power.("Best Score")==0,:)=[];
Record_SCM_FGO_exp(Record_SCM_FGO_exp.("Best Score")==0,:)=[];
Record_SCM_FGO_entropy(Record_SCM_FGO_entropy.("Best Score")==0,:)=[];
%%
fprintf("============ statistics ============\n")
num_valid_images = height(Record_SCM_FGO_entropy);
fprintf("num of valid images: %d\n",num_valid_images);
fprintf("num of re-localized images (rot err < 10 degrees):\n")
fprintf("CM_FGO: %d \n",length(find(Record_CM_FGO.("Max Rot Err")<10)))
fprintf("SCM_FGO_power: %d \n",length(find(Record_SCM_FGO_power.("Max Rot Err")<10)))
fprintf("SCM_FGO_exp: %d \n",length(find(Record_SCM_FGO_exp.("Max Rot Err")<10)))
fprintf("SCM_FGO_entropy: %d \n",length(find(Record_SCM_FGO_entropy.("Max Rot Err")<10)))
output_filename= "./matlab/Experiments/records/"+dataset_idx+"_rotation_record.mat";
save(output_filename);
%%
fprintf("============ time statistics ============\n")
fprintf("CM_FGO: %f,%f,%f\n",quantile(Record_CM_FGO.("Time"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_power: %f,%f,%f\n",quantile(Record_SCM_FGO_power.("Time"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_exp: %f,%f,%f\n",quantile(Record_SCM_FGO_exp.("Time"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_SCM_FGO_entropy.("Time"),[0.25,0.5,0.75]))

fprintf("============ max rot err statistics ============\n")
fprintf("Image Retriveal:%f,%f,%f\n",quantile(Record_SCM_FGO_entropy.("IR Err Rot"),[0.25,0.5,0.75]))
fprintf("CM_FGO: %f,%f,%f\n",quantile(Record_CM_FGO.("Max Rot Err"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_power: %f,%f,%f\n",quantile(Record_SCM_FGO_power.("Max Rot Err"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_exp: %f,%f,%f\n",quantile(Record_SCM_FGO_exp.("Max Rot Err"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_SCM_FGO_entropy.("Max Rot Err"),[0.25,0.5,0.75]))

fprintf("============ Recall at 3/5/10 degrees============\n")
fprintf("Image Retriveal: %f,%f,%f\n",sum(Record_SCM_FGO_entropy.("IR Err Rot")<[3,5,10])/num_valid_images*100)
fprintf("CM_FGO: %f,%f,%f\n",sum(Record_CM_FGO.("Max Rot Err")<[3,5,10])/num_valid_images*100)
fprintf("SCM_FGO_power: %f,%f,%f\n",sum(Record_SCM_FGO_power.("Max Rot Err")<[3,5,10])/num_valid_images*100)
fprintf("SCM_FGO_exp: %f,%f,%f\n",sum(Record_SCM_FGO_exp.("Max Rot Err")<[3,5,10])/num_valid_images*100)
fprintf("SCM_FGO_entropy: %f,%f,%f\n",sum(Record_SCM_FGO_entropy.("Max Rot Err")<[3,5,10])/num_valid_images*100)