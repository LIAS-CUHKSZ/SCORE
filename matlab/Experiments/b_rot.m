%%%%
% Rotation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% FGO_PnL vs EGO_PnL

%%% Author:  Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT

clear;
clc;
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(1);
data_folder="csv_dataset/"+dataset_idx+"/";
load(data_folder+"lines3D.mat");
%%% statistics
total_img=1000;
column_names=...
    ["Image ID","# 2D lines","epsilon_r","Outlier Ratio","Max Rot Err","Min Rot Err","# Rot Candidates","Best Score","GT Score","Time","Rot Vec"];
columnTypes =...
    ["int32","int32","double","double","double","double","int32","double","double","double","cell"];
Record_CM_FGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_CM_EGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_EGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
%%%  params
prox_thres = 1*pi/180; % for clustering proximate stabbers
branch_reso = pi/512; % terminate bnb when branch size < branch_reso
sample_reso = pi/512; % resolution for interval analysis
%%
for num =0:2000
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_idx=num*10;     
    frame_id = sprintf("%06d",img_idx);
    if ~exist(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv",'file')     %%% read 2D line data of cur image
        continue
    end
    K_p=readmatrix(data_folder+"intrinsics/frame_"+frame_id+".csv");
    T_gt = readmatrix(data_folder+"poses/frame_"+frame_id+".csv");
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/frame_"+frame_id+".csv")+1; % retrived sub-map
    K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1]; % intrinsic matrix
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    % lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1) 
    lines2D = readmatrix(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv"); 
    lines2D(:,1:3)=lines2D(:,1:3)*K; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    
    % ---------------------------------------------------------------------
    % --- 2. check observability using ground truth matching ---
    with_match_idx = find(lines2D(:,9)>=0);
    M = length(with_match_idx);
    n_2D_gt = zeros(M,3);  v_3D_gt = zeros(M,3); residual_r = zeros(M,1);
    for i=1:M
        matched_idx = int32(lines2D(with_match_idx(i),9))+1;
        n = lines2D(with_match_idx(i),1:3);
        v = lines3D(matched_idx,4:6)-lines3D(matched_idx,1:3);
        v = v /norm(v);
        n_2D_gt(i,:)=n; v_3D_gt(i,:)=v';
        residual_r(i) = (R_gt*n')'*v';
    end
    % basically we call the same function with ground truth matching
    epsilon_r = max(lines2D(with_match_idx,10))+0.001;
    id_gt = 1:M; id_gt = id_gt';
    kernel_buff = ones(M,1);
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D_gt,v_3D_gt,id_gt,kernel_buff,...
        branch_reso*2,epsilon_r,sample_reso*2,prox_thres);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    if max_err > 30
        fprintf("image "+num2str(img_idx)+" is ambigious in rotation, skip.\n");
        continue
    end
    % ---------------------------------------------------------------------
    % --- 3. semantic matching and saturation function design ---
    fprintf(num2str(img_idx)+"\n")
    lines3D_sub = lines3D(retrived_3D_line_idx,:);
    [ids,n_2D,v_3D,~]=match_line(lines2D,lines3D_sub);  % match with clustered 3D lines
    num_2D_lines = size(lines2D,1);
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(ids==i);
    end
    kernel_buff_SCM = zeros(num_2D_lines,max(match_count));
    for i = 1:num_2D_lines
        kernel_buff_SCM(i,1)=1;
        for j = 2:match_count(i)
            kernel_buff_SCM(i,j)=log(j)-log(j-1);
        end
    end
    kernel_buff_SCM(:,2:end)=kernel_buff_SCM(:,2:end)/sum(sum(kernel_buff_SCM(:,2:end)));
    kernel_buff_CM=ones(num_2D_lines,max(match_count));
    gt_inliers_idx = find(abs(dot(R_gt'*v_3D',n_2D'))<=epsilon_r);
    gt_inliers_id = ids(gt_inliers_idx);

    % ---------------------------------------------------------------------
    % --- 4. estimate rotation with Consensus Maximization  ---
    gt_score = calculate_score(gt_inliers_id,kernel_buff_CM);
    % CM_FGO
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_CM,...
        branch_reso,epsilon_r,sample_reso,prox_thres);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_CM_FGO(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};
    % CM_EGO

    [R_opt_top,best_score,num_candidate,time,~,~] = Sat_RotEGO(n_2D,v_3D,ids,kernel_buff_CM,...
        branch_reso,epsilon_r,prox_thres);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_CM_EGO(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};

    % ---------------------------------------------------------------------
    % --- 5. estimate rotation with Saturated Consensus Maximization  ---
    gt_score = calculate_score(gt_inliers_id,kernel_buff_SCM);
    % SCM_FGO
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,kernel_buff_SCM,...
        branch_reso,epsilon_r,sample_reso,prox_thres);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_SCM_FGO(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};
    % SCM_EGO
    [R_opt_top,best_score,num_candidate,time,~,~] = Sat_RotEGO(n_2D,v_3D,ids,kernel_buff_SCM,...
        branch_reso,epsilon_r,prox_thres);
    [min_err,max_err,R_min,R_max]=min_max_rot_error(num_candidate,R_opt_top,R_gt);
    Record_SCM_EGO(num+1,:)={img_idx,size(lines2D,1),epsilon_r,1-M/size(n_2D,1),max_err,min_err,num_candidate,best_score,gt_score,time,{rotmat2vec3d(R_max)}};
end
Record_CM_FGO(Record_CM_FGO.("Best Score")==0,:)=[];
Record_CM_EGO(Record_CM_EGO.("Best Score")==0,:)=[];
Record_SCM_FGO(Record_SCM_FGO.("Best Score")==0,:)=[];
Record_SCM_EGO(Record_SCM_EGO.("Best Score")==0,:)=[];
%%
fprintf("============ statistics ============\n")
fprintf("num of valid images: %d\n",height(Record_SCM_FGO));
fprintf("num of re-localized images (rot err < 10 degrees):\n")
fprintf("CM_EGO: %d \n",length(find(Record_CM_EGO.("Max Rot Err")<10)))
fprintf("CM_FGO: %d \n",length(find(Record_CM_FGO.("Max Rot Err")<10)))
fprintf("SCM_FGO: %d \n",length(find(Record_SCM_FGO.("Max Rot Err")<10)))
fprintf("SCM_EGO: %d \n",length(find(Record_SCM_EGO.("Max Rot Err")<10)))
% Record_large_Error = Record_SCM_FGO(Record_SCM_FGO.("Max Rot Err")>10,:);
% output_filename= "./matlab/Experiments/records/"+dataset_idx+"_rotation_record.mat";
% save(output_filename);