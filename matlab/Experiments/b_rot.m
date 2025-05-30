%%%%
% Orientation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% FGOPnL vs EGOPnL

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

clear;
clc;
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(3);
data_folder="csv_dataset/"+dataset_idx+"/";
load(data_folder+"lines3D.mat");

%%% statistics
total_img=1000;
column_names=...
    ["Image ID","# 2D lines","epsilon_r","Outlier Ratio","Rot Err","# Rot Candidates","Best Score","GT Score","Time"];
columnTypes =...
    ["int32","int32","double","double","double","int32","double","double","double"];
Record_CM_FGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO_clustered     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_CM_EGO     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_EGO_clustered     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
%%
%%%  params
% kernel_SCM = @(x) x^(-9);  % as a close approximation to @(x) 1-(x>1)
kernel_SCM = @(x) 2^(-x+1);
trunc_num=length(lines3D);
kernel_buffer_SCM=zeros(trunc_num,1);
kernel_buffer_CM = 1:1:trunc_num; kernel_buffer_CM = kernel_buffer_CM';
for i=1:trunc_num
    kernel_buffer_SCM(i)=kernel_SCM(i);
end
prox_thres = 2*pi/180; % keep candidates which have a same score and not proximate to each other
verbose_flag=0; % verbose mode for BnB
mex_flag=1; % use matlab mex code for acceleration
branch_reso = pi/512; % terminate bnb when branch size <= branch_reso
sample_reso = pi/256; % resolution for interval analysis
% set threshold
line_num_thres=5; % minimal number of 2D lines required in the image
parfor num =0:total_img
    img_idx=num*10;     
    %%% read 2D line data of cur image
    frame_id = sprintf("%06d",img_idx);
    if ~exist(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv",'file')
        continue
    end
    K_p=readmatrix(data_folder+"intrinsics/frame_"+frame_id+".csv");
    K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(data_folder+"poses/frame_"+frame_id+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    % lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1) 
    lines2D = readmatrix(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv"); 
    lines2D = lines2D(lines2D(:,4)~=0,:);   % delete 2D line without a semantic label
    %%%%% note: annotating the line below leads to a more challenging problem setting 
    lines2D = lines2D(lines2D(:,9)>=0,:); % delete 2D line without a real matching
    %%%%%%
    lines2D(:,1:3)=lines2D(:,1:3)*K; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    M = size(lines2D,1);
    if M <=line_num_thres
        continue
    end
    n_2D_gt = zeros(M,3);  v_3D_gt = zeros(M,3); residual_r = zeros(M,1);
    for i=1:M
        matched_idx = int32(lines2D(i,9))+1;
        n = lines2D(i,1:3);
        v = lines3D(matched_idx,4:6)-lines3D(matched_idx,1:3);
        v = v /norm(v);
        n_2D_gt(i,:)=n; v_3D_gt(i,:)=v';
    end
    epsilon_r = max(lines2D(:,10))+0.001;
    %%% check ambiguity
    id_gt = 1:M; id_gt = id_gt';
    [ambiguity_flag,err,~] = check_rot_ambiguity(n_2D_gt,v_3D_gt,id_gt,...
        branch_reso,epsilon_r,sample_reso,prox_thres,0,mex_flag,R_gt);
    if ambiguity_flag
        fprintf("image "+num2str(img_idx)+" is ambigious in rotation, skip.\n");
        continue
    end
    fprintf(num2str(img_idx)+"\n")
    %%% match 2D and 3D lines using semnatic label
    [ids,n_2D,v_3D,~]=match_line(lines2D,lines3D);  % match with unclustered 3D lines 
    [ids_cluster,n_2D_cluster,v_3D_cluster,~]=match_line(lines2D,lines3D_cluster);  % match with clustered 3D lines 
    %%%%%%%%%%%%%%%%%%% Estimate Orientation %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%% CM %%%%%%%%%%%%%%%%%%%%
    % gt_inliers_idx = abs(dot(R_gt'*v_3D',n_2D'))<=epsilon_r;
    % gt_inliers_id = ids(gt_inliers_idx);
    % gt_score = calculate_score(gt_inliers_id,kernel_buffer_CM);
    % % CM_FGO
    % [R_opt_top,best_score,num_candidate,time,~,~] = ...
    %     Sat_RotFGO(n_2D,v_3D,ids,kernel_buffer_CM,...
    %     branch_reso,epsilon_r,sample_reso,prox_thres,verbose_flag,mex_flag);
    % [min_err,~]=min_error(num_candidate,R_opt_top,R_gt);
    % Record_CM_FGO(num+1,:)={img_idx,M,epsilon_r,1-M/size(n_2D,1),min_err,num_candidate,best_score,gt_score,time};
    % % CM_EGO
    % [R_opt_top,best_score,num_candidate,time,~,~] = ...
    %     Sat_RotEGO(n_2D,v_3D,ids,kernel_buffer_CM,...
    %     branch_reso,epsilon_r,prox_thres,verbose_flag);
    % [min_err,~]=min_error(num_candidate,R_opt_top,R_gt);
    % Record_CM_EGO(num+1,:)={img_idx,M,epsilon_r,1-M/size(n_2D,1),min_err,num_candidate,best_score,gt_score,time};
    % %%%%%%%%%%%%%%%%%% SCM%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%%%%%%%%%%%% unclustered %%%%%%%%%%%%%%%%%%%%
    % % SCM_FGO
    % [R_opt_top,best_score,num_candidate,time,~,~] = ...
    %     Sat_RotFGO(n_2D,v_3D,ids,kernel_buffer_SCM,...
    %     branch_reso,epsilon_r,sample_reso,prox_thres,verbose_flag,mex_flag);
    % [min_err,~]=min_error(num_candidate,R_opt_top,R_gt);
    % Record_SCM_FGO(num+1,:)={img_idx,M,epsilon_r,1-M/size(n_2D,1),min_err,num_candidate,best_score,gt_score,time};
    %%%%%%%%%%%%%%%%%%% clustered %%%%%%%%%%%%%%%%%%%%%%
    gt_inliers_idx = find(abs(dot(R_gt'*v_3D_cluster',n_2D_cluster'))<=epsilon_r);
    gt_inliers_id = ids_cluster(gt_inliers_idx);
    gt_score = calculate_score(gt_inliers_id,kernel_buffer_SCM);
    % SCM_FGO_cluster
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D_cluster,v_3D_cluster,ids_cluster,kernel_buffer_SCM,...
        branch_reso,epsilon_r,sample_reso,prox_thres,verbose_flag,mex_flag);
    [min_err,~]=min_error(num_candidate,R_opt_top,R_gt);
    Record_SCM_FGO_clustered(num+1,:)={img_idx,M,epsilon_r,1-M/size(n_2D_cluster,1),min_err,num_candidate,best_score,gt_score,time};
    % % SCM_EGO_cluster
    % [R_opt_top,best_score,num_candidate,time,~,~] = ...
    % Sat_RotEGO(n_2D_cluster,v_3D_cluster,ids_cluster,kernel_buffer_SCM,...
    % branch_reso,epsilon_r,prox_thres,verbose_flag);
    % [min_err,~]=min_error(num_candidate,R_opt_top,R_gt);
    % Record_SCM_EGO_clustered(num+1,:)={img_idx,M,epsilon_r,1-M/size(n_2D_cluster,1),min_err,num_candidate,best_score,gt_score,time};
end
Record_CM_FGO(Record_CM_FGO.("Best Score")==0,:)=[];
Record_CM_EGO(Record_CM_EGO.("Best Score")==0,:)=[];
Record_SCM_FGO(Record_SCM_FGO.("Best Score")==0,:)=[];
Record_SCM_FGO_clustered(Record_SCM_FGO_clustered.("Best Score")==0,:)=[];
Record_SCM_EGO_clustered(Record_SCM_EGO_clustered.("Best Score")==0,:)=[];
%%
fprintf("============ statistics ============\n")
fprintf("num of valid images: %d\n",height(Record_SCM_FGO_clustered));
fprintf("num of re-localized images (rot err < 30 degrees):\n")
fprintf("CM_EGO: %d \n",length(find(Record_CM_EGO.("Rot Err")<30)))
fprintf("CM_FGO: %d \n",length(find(Record_CM_FGO.("Rot Err")<30)))
fprintf("SCM_FGO: %d \n",length(find(Record_SCM_FGO.("Rot Err")<30)))
fprintf("SCM_FGO_clustered: %d \n",length(find(Record_SCM_FGO_clustered.("Rot Err")<30)))
fprintf("SCM_EGO_clustered: %d \n",length(find(Record_SCM_EGO_clustered.("Rot Err")<30)))
output_filename= "./matlab/Experiments/records/"+dataset_idx+"_rotation_record.mat";
% save(output_filename);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function []=plot_bound_record(L_record,U_record)
    plot(1:length(U_record),U_record,'Color','b')
    hold on
    plot(1:length(L_record),L_record,'Color','b')
    hold off
end

function score=calculate_score(inlier_ids,kernel_buffer)
    trunc_num = length(kernel_buffer);
    score=0;
    unique_ids=unique(inlier_ids);
    for i=1:length(unique_ids)
        num = sum(inlier_ids==unique_ids(i));
        num = min(num,trunc_num);
        for j=1:num
            score=score+kernel_buffer(j);
        end
    end
end
