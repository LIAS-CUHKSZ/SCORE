%%%%
% Rotation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% --- Note!! ---
% If you don't want to or can't use the compiled mex functions,
% remeber to set variables 'mex_flag=0' in function Sat_RotFGO

%%% Author:  Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
clear;
clc
scene_idx = 1; % choose one scene
pred_flag = 1; % set 1 if use predicted semantic label
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
dataset_name = dataset_names(scene_idx);

% load semantic remapping 
if pred_flag
    data_folder="csv_dataset/"+dataset_name+"_pred/";
    remapping = load(data_folder+"remapping.txt");
else
    data_folder="csv_dataset/"+dataset_name+"/";
    remapping = [];
end

% params
prox_thres = 1*pi/180; % for clustering proximate stabbers
branch_reso = pi/512; % terminate bnb early when branch size < branch_reso
sample_reso = pi/512; % resolution for interval analysis
epsilon_r = 0.015;
L_list = [30,60,120,240,360,480,600]; % q from 0.3 to 0.9
num_q = length(L_list);

% table to record results
total_img=2000;
column_names=...
    ["Image ID","# 2D lines","epsilon_r","Outlier Ratio","IR Err Rot","Max Rot Err","Min Rot Err","# Rot Candidates","Best Score","GT Score","Time","Rot Candidates"];
columnTypes =...
    ["int32","int32","double","double","double","double","double","int32","double","double","double","cell"];
Record_CM_FGO      =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_trunc   =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
temp_buffer = cell(total_img,num_q);  % store results for likelihood-based saturation function in a temp buffer
%%
lines3D=readmatrix(data_folder+"/3Dlines.csv");
parfor num = 1:total_img
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_idx=num*10;
    frame_id = sprintf("%06d",img_idx);
    if ~exist(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv",'file')    
        continue
    end

    % load 2D data
    K_p=readmatrix(data_folder+"intrinsics/frame_"+frame_id+".csv");
    T_gt = readmatrix(data_folder+"poses/frame_"+frame_id+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1]; % intrinsic matrix
    % lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1)
    lines2D = readmatrix(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv");
    lines2D(lines2D(:,4)==0,:)=[]; % delete lines without semantic label
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    
    % load image retrivel results and prune search space
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/frame_"+frame_id+".csv")+1; % retrived sub-map
    lines3D_sub = lines3D(retrived_3D_line_idx,:);
    retrived_closest_pose = readmatrix(data_folder+"retrived_closest_pose/frame_"+frame_id+".csv"); % pose of the most similar retrived image
    retrived_err_rot = angular_distance(retrived_closest_pose(1:3,1:3),R_gt);
    [alpha,phi,theta] = rot2angle(retrived_closest_pose(1:3,1:3)');
    if abs(phi) < 3*pi/180    % ambiguous case
        west_east_flag = 2;
    else
        if phi<0
            west_east_flag = 1;
        else
            west_east_flag = 0;
        end
    end

    % ---------------------------------------------------------------------
    % --- 2. semantic matching and calculate outlier ratio ---
    [lines2D,lines3D_sub]=remap_semantic_id(lines2D,lines3D_sub,remapping);
    [ids,n_2D,v_3D,~]=match_line(lines2D,lines3D_sub);

    % skip image with too few matched 2D lines due to observability issue
    if length(unique(ids)) < 5
        fprintf("image "+num2str(img_idx)+" has less than 5 lines, skip.\n");
        continue
    end
    total_match_num = size(n_2D,1);
    with_match_ids = find(lines2D(:,9)>0);
    if pred_flag
        predicted_semantics = lines2D(with_match_ids,4);
        true_semantics = lines3D(lines2D(with_match_ids,9),7);
        outlier_ratio = 1-nnz(predicted_semantics==true_semantics)/total_match_num;
    else
        outlier_ratio = 1-length(with_match_ids)/total_match_num;
    end

    % --- 3. saturation function design ---
    num_2D_lines = size(lines2D,1);
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(ids==i);
    end
    L = sum(log(match_count(match_count>0)));  % a sufficiently large number
    kernel_buff_CM = ones(num_2D_lines,max(match_count));
    kernel_buff_SCM_trunc = zeros(num_2D_lines,max(match_count));
    kernel_buff_SCM_trunc(:,1)=1;
    kernel_buff_SCM_power = zeros(num_2D_lines,max(match_count));
    % likelihood-based saturation functions with different parameter choice
    kernel_buff_SCM_ML_lists = cell(num_q,1);
    for k = 1:num_q
        kernel_buff_SCM_ML_lists{k} = zeros(num_2D_lines,max(match_count));
    end
    for i = 1:num_2D_lines
        if match_count(i)==0
            continue
        end
        for j =1:match_count(i)
            kernel_buff_SCM_power(i,j)=j^(-8);
            for k = 1:num_q
                kernel_buff_SCM_ML_lists{k}(i,j) = log(1+L_list(k)*j/match_count(i))-log(1+L_list(k)*(j-1)/match_count(i));
            end
        end
    end

    % ---------------------------------------------------------------------
    % --- 4. rotation estimation starts here ---
    fprintf(num2str(img_idx)+"\n")
    % find inliers under ground truth rotation
    gt_inliers_idx = find(abs(dot(R_gt'*v_3D',n_2D'))<=epsilon_r);
    gt_inliers_id = ids(gt_inliers_idx);

    % % CM_FGO
    % gt_score = calculate_score(gt_inliers_id,kernel_buff_CM);
    % [R_opt,best_score,num_candidate,time,~,~] = ...
    %     Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_CM,...
    %     branch_reso,epsilon_r,sample_reso,prox_thres,west_east_flag);
    % [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt,R_gt);
    % Record_CM_FGO(num,:)={img_idx,size(lines2D,1),epsilon_r,outlier_ratio,retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{R_opt}};

    % SCM_FGO_trunc
    gt_score = calculate_score(gt_inliers_id,kernel_buff_SCM_trunc);
    [R_opt,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_SCM_trunc,...
        branch_reso,epsilon_r,sample_reso,prox_thres,west_east_flag);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt,R_gt);
    Record_SCM_trunc(num,:)={img_idx,size(lines2D,1),epsilon_r,outlier_ratio,retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{R_opt}};

    % SCM_FGO_ML
    for k = 1:num_q
        gt_score = calculate_score(gt_inliers_id,kernel_buff_SCM_ML_lists{k});
        [R_opt,best_score,num_candidate,time,~,~] = ...
            Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_SCM_ML_lists{k},...
            branch_reso,epsilon_r,sample_reso,prox_thres,west_east_flag);

        [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt,R_gt);
        temp_buffer{num,k}={img_idx,size(lines2D,1),epsilon_r,outlier_ratio,retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{R_opt}};
    end
end
%%
% organize data table
Record_CM_FGO(Record_CM_FGO.("Best Score")==0,:)=[];
Record_SCM_trunc(Record_SCM_trunc.("Best Score")==0,:)=[];

% copy data in the temp buffer
Record_SCM_ML_lists = cell(num_q,1);
for k = 1:num_q
    Record_SCM_ML_lists{k} =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
end
for num=1:total_img
    for k=1:num_q
        temp_result = temp_buffer{num,k};
        if ~isempty(temp_result)
             Record_SCM_ML_lists{k}(num,:)=temp_result;
        end
    end
end
for k=1:num_q
    Record_SCM_ML_lists{k}(Record_SCM_ML_lists{k}.("Best Score")==0,:)=[];
end
% save data
if pred_flag
    output_filename= "./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record.mat";
else
    output_filename= "./matlab/Experiments/records/gt_semantics/"+dataset_name+"_rotation_record.mat";
end
save(output_filename);
%%
% print useful statisticss
num_valid_images = height(Record_CM_FGO);
fprintf("============ outlier ratio ============\n")
fprintf("Quantiles of outlier ratio:%f,%f,%f\n", quantile(Record_CM_FGO.("Outlier Ratio"),[0.25,0.5,0.75]));
%
fprintf("============ number of relocalized image ============\n")
fprintf("num of valid images: %d\n",num_valid_images);
fprintf("num of re-localized images (rot err < 10 degrees):\n")
fprintf("CM_FGO: %d \n",length(find(Record_CM_FGO.("Max Rot Err")<10)))
fprintf("SCM_FGO(trunc): %d \n",length(find(Record_SCM_trunc.("Max Rot Err")<10)))
const = epsilon_r/1;
for k = 1:num_q
    fprintf("SCM_FGO(q=%f,L=%f): %d \n", const*L_list(k)/(1+const*L_list(k)), L_list(k),length(find(Record_SCM_ML_lists{k}.("Max Rot Err")<10)))
end

% fprintf("============ time statistics ============\n")
% fprintf("CM_FGO: %f,%f,%f\n",quantile(Record_CM_FGO.("Time"),[0.25,0.5,0.75]))
% fprintf("SCM_FGO(trunc): %f,%f,%f\n",quantile(Record_SCM_trunc.("Time"),[0.25,0.5,0.75]))
% for k = 1:num_q
%     fprintf("SCM_FGO(q=%f,L=%f): %f,%f,%f\n", const*L_list(k)/(1+const*L_list(k)),L_list(k),quantile(Record_SCM_ML_lists{k}.("Time"),[0.25,0.5,0.75]))
% end

fprintf("============ max rot err statistics ============\n")
fprintf("Image Retriveal:%f,%f,%f\n",quantile(Record_CM_FGO.("IR Err Rot"),[0.25,0.5,0.75]))
fprintf("CM_FGO: %f,%f,%f\n",quantile(Record_CM_FGO.("Max Rot Err"),[0.25,0.5,0.75]))
fprintf("SCM_FGO(trunc): %f,%f,%f\n",quantile(Record_SCM_trunc.("Max Rot Err"),[0.25,0.5,0.75]))
for k = 1:num_q
    fprintf("SCM_FGO(q=%f,L=%f): %f,%f,%f\n", const*L_list(k)/(1+const*L_list(k)),L_list(k),quantile(Record_SCM_ML_lists{k}.("Max Rot Err"),[0.25,0.5,0.75]))
end
%
fprintf("============ Recall at 3/5/10 degrees============\n")
fprintf("Image Retriveal: %f,%f,%f\n",sum(Record_CM_FGO.("IR Err Rot")<[3,5,10])/num_valid_images*100)
fprintf("CM_FGO: %f,%f,%f\n",sum(Record_CM_FGO.("Max Rot Err")<[3,5,10])/num_valid_images*100)
fprintf("SCM_FGO(trunc): %f,%f,%f\n",sum(Record_SCM_trunc.("Max Rot Err")<[3,5,10])/num_valid_images*100)
for k = 1:num_q
    fprintf("SCM_FGO(q=%f,L=%f): %f,%f,%f\n", const*L_list(k)/(1+const*L_list(k)),L_list(k),sum(Record_SCM_ML_lists{k}.("Max Rot Err")<[3,5,10])/num_valid_images*100)
end