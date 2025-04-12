%%%%
% Orientation Estimation
% FGO vs EGO
% Saturated Consensus Maximization vs Consensus Maximization

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%
%%
clear;
clc;
% candidate saturation functions
kernel_1 = @(x) exp(-x+1);
kernel_2 = @(x) x^-2;
kernel_3 = @(x) x^-8;
kernel_4 = @(x) 2^(-x+1)/x;
kernel_5 = @(x) 1/factorial(x);
kernel_6 = @(x) 2^(-x+1);
kernel_7 = @(x) 2/x/(x+1);
%%
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(1);
output_filename= "matlab/Experiments/record/"+dataset_idx+"_rotation_record.mat";
folder_name="csv_dataset/"+dataset_idx+"/";
load(folder_name+"lines3D.mat");
numRows=11000;
column_names=["Image Index","Time","Orient Err","# 2D lines with match","Score","Score under gt","# candidates"];
columnTypes = ["int32","double","double","int32","double","double","int32"];
Record_SCM_FGO_clustered     =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO_unclustered   =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
% Record_CM_FGO =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
% Record_SCM_EGO_clustered    =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
% Record_CM_EGO=table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
%%%  params
kernel = kernel_3;
trunc_num=100;
kernel_buffer=zeros(trunc_num,1);
for i=1:trunc_num
    kernel_buffer(i)=kernel(i);
end
line_num_thres=20; % minimal number of 2D lines required in the image
total_picture=100; 
% rotation bnb
verbose_flag=0; % verbose mode for BnB
mex_flag=1; % use mex code to further accelerate BnB
branch_reso = pi/1024;
sample_reso = pi/512;
parfor num =60:100
    num
    %%% read 2D line data of cur image
    frame_id = sprintf("%06d",num*10);
    if ~exist(folder_name+"lines2d\frame_"+frame_id+"2dlines.csv",'file')
        continue
    end
    % lines2D(Nx10): normal vector(3x1), semantic label(1), projection error(2x1), endpoint a(2x1), endpoint b(2x1) 
    lines2D = readmatrix(folder_name+"lines2d\frame_"+frame_id+"2dlines.csv"); 
    lines2D = lines2D(lines2D(:,4)~=0,:); % delete 2D line without a semantic label
    if length(lines2D)<line_num_thres
        continue
    end
    epsilon_r=max(lines2D(:,6)*1.25);
    K_p=readmatrix(folder_name+"intrinsics\frame_"+frame_id+".csv");
    K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(folder_name+"poses\frame_"+frame_id+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    lines2D(:,1:3)=lines2D(:,1:3)*K; 
    lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    %%% match 2D and 3D lines using semnatic label
    % match with unclustered 3D lines 
    [data_2D_n,data_3D_v,id]=match_line(lines2D,lines3D); 
    % fprintf("# match with all 3D lines:       %d\n",length(id));
    % match with clustered 3D lines 
    [data_2D_n_clustered,data_3D_v_clustered,id_cluster]=match_line(lines2D,lines3D_cluster);
    % fprintf("# match with clustered 3D lines: %d\n",length(id_cluster)); 
    clustered_match_num=0;
    for i=1:size(lines2D,1)
        idx_matched_3D = find(abs(lines3D_cluster(:,7)-lines2D(i,4))<0.1);
        clustered_match_num = clustered_match_num+length(idx_matched_3D);
    end
    %%%%%%%%%%%%%%%%%%% Estimate Orientation %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gt_inliers_idx = find(abs(dot(R_gt'*data_3D_v_clustered',data_2D_n_clustered'))<=epsilon_r);
    gt_inliers_id = id_cluster(gt_inliers_idx);
    gt_score = calculate_score(gt_inliers_id,kernel_buffer);
    num_2D_line_match=length(unique(id_cluster));
    % Sat_FGO_clustered
    [R_opt_top,best_score,num_candidate,time,~,~] = Sat_RotFGO(data_2D_n_clustered,data_3D_v_clustered,id_cluster,kernel_buffer,branch_reso,epsilon_r,sample_reso,verbose_flag,mex_flag);
    line_pair_data = data_process(data_2D_n_clustered,data_3D_v_clustered);
    %%%%%%%%%%%%%%%%%%%%% Acclerated BnB %%%%%%%%%%%%%%%%%%%%%
    %%% Initialize the BnB process
    % calculate bounds for east and west semispheres.
    [min_err,R_opt]=min_error(num_candidate,R_opt_top,R_gt);
    est_inliers_idx=find(abs(dot(R_opt'*data_3D_v_clustered',data_2D_n_clustered'))<=epsilon_r);
    est_inliers_id = id_cluster(est_inliers_idx);
    Record_SCM_FGO_clustered(num+1,:)={num*10,time,min_err,num_2D_line_match,best_score,gt_score,num_candidate};
end
% save(output_filename,"Record_CM_EGO","Record_CM_FGO","Record_SCM_FGO_unclustered","Record_SCM_EGO_clustered", ...
%                             "Record_SCM_FGO_clustered","sampleSize","sample_resolution","epsilon_r","branch_resolution");
function []=plot_bound_record(L_record,U_record)
    plot(1:length(U_record),U_record,'Color','b')
    hold on
    plot(1:length(L_record),L_record,'Color','b')
    hold off
end

function score=calculate_score(inlier_ids,kernel_buffer)
    score=0;
    unique_ids=unique(inlier_ids);
    for i=1:length(unique_ids)
        num = sum(inlier_ids==unique_ids(i));
        for j=1:num
            score=score+kernel_buffer(j);
        end
    end
end

