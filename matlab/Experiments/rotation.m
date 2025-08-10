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
dataset_names = ["S1","S2","S3","S4"];

% configure setting
scene_idx = 2; dataset_name=dataset_names(scene_idx); % which scene
pred_flag = 0; % use predicted label?
two_or_eight = 0; % side length = pi or pi/2?

% set rotation params
branch_reso = pi/512;     % terminate bnb early when branch size < branch_reso
sample_reso = pi/256;     % resolution for interval analysis
prox_thres = branch_reso; % for clustering proximate stabbers
epsilon_r = 0.015;        % rotation residual tolerance
u_r = 1;                  % rotation residual upper bound
if pred_flag
    q_list = [0.2,0.35,0.5,0.65,0.8];
    C_list = q_list./(1-q_list)*u_r/epsilon_r;
    data_folder="csv_dataset/"+dataset_name+"_pred/";
    remapping = load(data_folder+"remapping.txt"); % label remapping for predicted results
else
    q_list = [0.6,0.7,0.8,0.9,0.99];
    C_list = q_list./(1-q_list)*u_r/epsilon_r;
    data_folder="csv_dataset/"+dataset_name+"/";
    remapping = [];
end
num_q = length(C_list);

% load query image list
query_list_file = data_folder+"query.txt";
fid = fopen(query_list_file, 'r'); % 'r' for read access
tline = fgetl(fid); % Read the first line
query_list=strings(500,1);
num_images = 0;
while ischar(tline) % Loop while tline is a character array (not -1)
    num_images = num_images + 1;
    query_list(num_images) = tline(1:end-4); 
    % Process the current line (tline) here
    tline = fgetl(fid); % Read the next line
end
query_list(num_images+1:end)=[];

% record table 
column_names=...
    ["Image Name","# 2D lines","epsilon_r","Outlier Ratio","IR Err Rot","Max Rot Err","Min Rot Err","# Rot Candidates","Best Score","GT Score","Time","Rot Candidates"];
columnTypes =...
    ["string","int32","double","double","double","double","double","int32","double","double","double","cell"];
Record_CM_FGO      =table('Size', [num_images, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_trunc   =table('Size', [num_images, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);

% go through each query image
temp_buffer = cell(num_images,num_q);  % store results for likelihood-based saturation function in a temp buffer
lines3D=readmatrix(data_folder+"/3Dlines.csv");
for num = 1:num_images
% parfor num = 1:num_images % use parfor to run in parallel
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_name = query_list(num);
    % load 2D data
    K_p=readmatrix(data_folder+"intrinsics/"+img_name+".csv");
    T_gt = readmatrix(data_folder+"poses/"+img_name+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);

    % load image retrivel results and prune search space
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1]; % intrinsic matrix
    % lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1)
    lines2D = readmatrix(data_folder+"lines2D/"+img_name+"_2Dlines.csv");
    lines2D(lines2D(:,4)==0,:)=[]; % delete lines without semantic label
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';

    % load image retrivel results and prune search space
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/"+img_name+".csv")+1; % retrived sub-map
    lines3D_sub = lines3D(retrived_3D_line_idx,:);
    retrived_closest_pose = readmatrix(data_folder+"retrived_closest_pose/"+img_name+"_retrived_pose.csv"); % pose of the most similar retrived image
    retrived_err_rot = angular_distance(retrived_closest_pose(1:3,1:3),R_gt);
    [alpha,phi,theta] = rot2angle(retrived_closest_pose(1:3,1:3)');
    if two_or_eight
        side_length = pi;
    else
        side_length = pi/2;
    end
    delta = 3*pi/180; % delta defines the size of an amiguous region
    initial_branch = confine_sphere(alpha,phi,side_length,delta);
    % ---------------------------------------------------------------------
    % --- 2. semantic matching and calculate outlier ratio ---
    [lines2D,lines3D_sub]=remap_semantic_id(lines2D,lines3D_sub,remapping);
    [ids,n_2D,v_3D,~]=match_line(lines2D,lines3D_sub);

    % skip image with too few matched 2D lines due to observability issue
    if length(unique(ids)) < 5
        fprintf(img_name+" has less than 5 lines, skip.\n");
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
    sat_buff_CM = ones(num_2D_lines,max(match_count));
    sat_buff_SCM_trunc = zeros(num_2D_lines,max(match_count));
    sat_buff_SCM_trunc(:,1)=1;
    % likelihood-based saturation functions with different parameter choice
    sat_buff_SCM_ML_lists = cell(num_q,1);
    for k = 1:num_q
        sat_buff_SCM_ML_lists{k} = zeros(num_2D_lines,max(match_count));
    end
    for i = 1:num_2D_lines
        if match_count(i)==0
            continue
        end
        for j =1:match_count(i)
            for k = 1:num_q
                sat_buff_SCM_ML_lists{k}(i,j) = log(1+C_list(k)*j/match_count(i))-log(1+C_list(k)*(j-1)/match_count(i));
            end
        end
    end

    % ---------------------------------------------------------------------
    % --- 4. rotation estimation starts here ---
    fprintf(img_name+"\n")
    % find inliers under ground truth rotation
    gt_inliers_idx = find(abs(dot(R_gt'*v_3D',n_2D'))<=epsilon_r);
    gt_inliers_id = ids(gt_inliers_idx);

    % CM_FGO
    gt_score = calculate_score(gt_inliers_id,sat_buff_CM);
    [R_opt,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,sat_buff_CM,...
        branch_reso,epsilon_r,sample_reso,prox_thres,initial_branch);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt,R_gt);
    Record_CM_FGO(num,:)={img_name,size(lines2D,1),epsilon_r,outlier_ratio,retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{R_opt}};

    % SCM_FGO_trunc
    gt_score = calculate_score(gt_inliers_id,sat_buff_SCM_trunc);
    [R_opt,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,sat_buff_SCM_trunc,...
        branch_reso,epsilon_r,sample_reso,prox_thres,initial_branch);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt,R_gt);
    Record_SCM_trunc(num,:)={img_name,size(lines2D,1),epsilon_r,outlier_ratio,retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{R_opt}};

    % SCM_FGO_ML
    for k = 1:num_q
        gt_score = calculate_score(gt_inliers_id,sat_buff_SCM_ML_lists{k});
        [R_opt,best_score,num_candidate,time,~,~] = ...
            Sat_RotFGO(n_2D,v_3D,ids,sat_buff_SCM_ML_lists{k},branch_reso,epsilon_r,sample_reso,prox_thres,initial_branch);

        [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt,R_gt);
        temp_buffer{num,k}={img_name,size(lines2D,1),epsilon_r,outlier_ratio,retrived_err_rot,max_err,min_err,num_candidate,best_score,gt_score,time,{R_opt}};
    end
end
%%
% delete void record
void_idx = find(Record_CM_FGO.epsilon_r==0);
% copy data in the temp buffer
Record_SCM_ML_lists = cell(num_q,1);
for k = 1:num_q
    Record_SCM_ML_lists{k} =table('Size', [num_images, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
end
for num=1:num_images
    if Record_CM_FGO.epsilon_r(num) == 0
        continue
    end
    for k=1:num_q
        Record_SCM_ML_lists{k}(num,:) = temp_buffer{num,k};
    end
end
%
Record_CM_FGO(void_idx,:)=[];
Record_SCM_trunc(void_idx,:)=[];
for k=1:num_q
    Record_SCM_ML_lists{k}(void_idx,:)=[];
end

%% output
if pred_flag
    output_folder = "./matlab/Experiments/records/pred_semantics/";
    if two_or_eight
        output_filename= output_folder+dataset_name+"_pred_rotation_record_2.mat";
    else
        output_filename= output_folder+dataset_name+"_pred_rotation_record_8.mat";
    end
else
    output_folder = "./matlab/Experiments/records/gt_semantics/";
    if two_or_eight
        output_filename= output_folder+dataset_name+"_rotation_record_2.mat";
    else
        output_filename= output_folder+dataset_name+"_rotation_record_8.mat";
    end
end
if ~exist(output_folder,'dir') 
    mkdir(output_folder); 
end
save(output_filename);

% print useful statisticss
num_valid_images = height(Record_SCM_ML_lists{1});
fprintf("============ outlier ratio ============\n")
fprintf("Quantiles of outlier ratio:%f,%f,%f\n", quantile(Record_SCM_ML_lists{1}.("Outlier Ratio"),[0.25,0.5,0.75]));
%
fprintf("============ number of relocalized image ============\n")
fprintf("num of valid images: %d\n",num_valid_images);
fprintf("num of re-localized images (rot err < 10 degrees):\n")
fprintf("CM_FGO: %d \n",length(find(Record_CM_FGO.("Max Rot Err")<10)))
fprintf("SCM_FGO(trunc): %d \n",length(find(Record_SCM_trunc.("Max Rot Err")<10)))
for k = 1:num_q
    fprintf("SCM_FGO(q=%f,C=%f): %d \n", q_list(k), C_list(k),length(find(Record_SCM_ML_lists{k}.("Max Rot Err")<10)))
end

fprintf("============ time quantile ============\n")
fprintf("CM_FGO: %f,%f,%f\n",quantile(Record_CM_FGO.("Time"),[0.25,0.5,0.75]))
fprintf("SCM_FGO(trunc): %f,%f,%f\n",quantile(Record_SCM_trunc.("Time"),[0.25,0.5,0.75]))
for k = 1:num_q
    fprintf("SCM_FGO(q=%f,C=%f): %f,%f,%f\n", q_list(k),C_list(k),quantile(Record_SCM_ML_lists{k}.("Time"),[0.25,0.5,0.75]))
end

fprintf("============ max rot err quantile ============\n")
fprintf("Image Retriveal:%f,%f,%f\n",quantile(Record_CM_FGO.("IR Err Rot"),[0.25,0.5,0.75]))
fprintf("CM_FGO: %f,%f,%f\n",quantile(Record_CM_FGO.("Max Rot Err"),[0.25,0.5,0.75]))
fprintf("SCM_FGO(trunc): %f,%f,%f\n",quantile(Record_SCM_trunc.("Max Rot Err"),[0.25,0.5,0.75]))
for k = 1:num_q
    fprintf("SCM_FGO(q=%f,C=%f): %f,%f,%f\n", q_list(k),C_list(k),quantile(Record_SCM_ML_lists{k}.("Max Rot Err"),[0.25,0.5,0.75]))
end
%
fprintf("============ Recall at 3/5/10 degrees============\n")
fprintf("Image Retriveal: %f,%f,%f\n",sum(Record_CM_FGO.("IR Err Rot")<[3,5,10])/num_valid_images*100)
fprintf("CM_FGO: %f,%f,%f\n",sum(Record_CM_FGO.("Max Rot Err")<[3,5,10])/num_valid_images*100)
fprintf("SCM_FGO(trunc): %f,%f,%f\n",sum(Record_SCM_trunc.("Max Rot Err")<[3,5,10])/num_valid_images*100)
for k = 1:num_q
    fprintf("SCM_FGO(q=%f,C=%f): %f,%f,%f\n", q_list(k),C_list(k),sum(Record_SCM_ML_lists{k}.("Max Rot Err")<[3,5,10])/num_valid_images*100)
end
