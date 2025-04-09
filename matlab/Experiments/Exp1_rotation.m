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
kernel = kernel_3;
%%
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(1);
output_filename= "matlab/Experiments/record/"+dataset_idx+"_rotation_record.mat";
folder_name="csv_dataset/"+dataset_idx+"/";
load(folder_name+"lines3d.mat");
numRows=11000;
column_names=["Image Index","Angular Error","Translation Error","Final # Inliers","BnB time","# Rot Candidates",...
            "# line pairs","# gt matches","# inliers under est","best lower bound","2d line label ratio","alpha_opt","phi_opt","theta_opt"];
columnTypes = ["int32","double","double","int32","double","int32","int32","int32","int32","double","double","double","double","double"];
Record_SCM_FGO_clustered     =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_EGO_clustered    =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO_unclustered   =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_CM_FGO =table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_CM_EGO=table('Size', [numRows, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
%%%  params
% rotation bnb
verbose_flag=0;
epsilon_r = 0.0125;
branch_resolution = pi/512;
sample_resolution = pi/256;
% translation ransac
sampleSize=4;
total_picture=1000;
line_num_thres=20;
tic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  for  %%%%%%%%%%%%%%%%%
parfor num =0:total_picture
    num
    
    frame_id = sprintf("%06d",num*10);
    if ~exist(datasetname+"lines2d\frame_"+frame_id+"2dlines.csv",'file')
        continue
    end
    lines2d = readmatrix(datasetname+"lines2d\frame_"+frame_id+"2dlines.csv");
    lines2d = lines2d(lines2d(:,4)~=0,:);
    if length(lines2d)<line_num_thres
        continue
    end
    semantic_label_mode = mode(lines2d(:,4));
    semantic_mode_ratio = sum(lines2d(:,4)==mode(lines2d(:,4)))/length(lines2d);
    K_p=readmatrix(datasetname+"intrinsics\frame_"+frame_id+".csv");
    K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(datasetname+"poses\frame_"+frame_id+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    lines2d(:,1:3)=lines2d(:,1:3)*K;
    lines2d(:,1:3)=lines2d(:,1:3)./vecnorm(lines2d(:,1:3)')';
    %%%%%%%%%%%%%%%%%%%%%%
    total_match_num=0;
    for i=1:size(lines2d,1)
        idx_matched_3d = find(abs(lines3d(:,7)-lines2d(i,4))<0.1);
        total_match_num = total_match_num+length(idx_matched_3d);
    end
    data_3d_v=zeros(total_match_num,3);
    data_2d_N=zeros(total_match_num,3);
    data_3d_c=zeros(total_match_num,3);
    id=zeros(total_match_num,1);
    %
    clustered_match_num=0;
    for i=1:size(lines2d,1)
        idx_matched_3d = find(abs(lines3d_cluster(:,7)-lines2d(i,4))<0.1);
        clustered_match_num = clustered_match_num+length(idx_matched_3d);
    end
    data_3d_v_clustered=zeros(clustered_match_num,3);
    data_2d_N_clustered=zeros(clustered_match_num,3);
    id_cluster=zeros(clustered_match_num,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%% clustered %%%%%%%%%%%%%%%%%%%%%%%%%%
    temp_cluster=0;
    temp=0;
    for i=1:size(lines2d,1)
        %%%
        idx_matched_3d = find(abs(lines3d_cluster(:,7)-lines2d(i,4))<0.1);
        num_matched=length(idx_matched_3d);
        for j = 1:num_matched
            data_2d_N_clustered(temp_cluster+j,:) = lines2d(i,1:3);
            data_3d_v_clustered(temp_cluster+j,:) = lines3d_cluster(idx_matched_3d(j),4:6);
            id_cluster(temp_cluster+j)=i;
        end
        temp_cluster=temp_cluster+num_matched;
        %%
        idx_matched_3d = find(abs(lines3d(:,7)-lines2d(i,4))<0.1);
        num_matched=length(idx_matched_3d);
        for j = 1:num_matched
            data_2d_N(temp+j,:) = lines2d(i,1:3);
            data_3d_v(temp+j,:) = lines3d(idx_matched_3d(j),4:6);
            data_3d_c(temp+j,:) = lines3d(idx_matched_3d(j),1:3);
            id(temp+j)=i;
        end
        temp=temp+num_matched;
    end
    gt_inliers = abs(dot(R_gt'*data_3d_v_clustered',data_2d_N_clustered'))<=epsilon_r;
    %%%%%%%%%%%%%%%%%%%%%%%  My_sat_clustered %%%%%%%%%%%%%%%%%%%%%%%
    %%%%%% rot
    % [R_sat_My_transpose,U_record_sat_My,L_record_sat_My,best_L_sat_My,num_candidate_sat_My,time_sat_My] =Sat_RotACM_My(data_3d_v_clustered,data_2d_N_clustered,branch_resolution ,epsilon_r,sample_resolution,verbose_flag,id_cluster,kernel);
    [R_sat_My_transpose,U_record_sat_My,L_record_sat_My,best_L_sat_My,num_candidate_sat_My,time_sat_My] =Sat_RotACM_My(data_3d_v_clustered,data_2d_N_clustered,branch_resolution ,epsilon_r,sample_resolution,verbose_flag,id_cluster);

    [min_error_sat_My,R_opt_sat_My]=min_error(num_candidate_sat_My,R_sat_My_transpose,R_gt);
    [alpha,phi,theta]=rot2angle(R_opt_sat_My);
    found_inliers_sat_My = abs(dot(R_opt_sat_My'*data_3d_v_clustered',data_2d_N_clustered'))<=epsilon_r;
    %%%%%%  result record
    Record_SCM_FGO_clustered(num+1,:)={num*10,min_error_sat_My,0,0,time_sat_My,num_candidate_sat_My,...
                            length(data_3d_v_clustered),length(unique(id_cluster(gt_inliers))),...
                            length(unique(id_cluster(found_inliers_sat_My))),best_L_sat_My,semantic_mode_ratio,alpha,phi,theta};

    %%%%%%%%%%%%%%%%%%%%%%%  My_sat_unclustered %%%%%%%%%%%%%%%%%%%%%
    %%%%%% rot
    % [R_sat_My_unclustered_transpose,U_record_sat_My_unclustered,L_record_sat_My_unclustered,best_L_sat_My_unclustered,num_candidate_sat_My_unclustered,time_sat_My_unclustered] =Sat_RotACM_My(data_3d_v,data_2d_N,branch_resolution ,epsilon_r,sample_resolution,verbose_flag,id,kernel);
    [R_sat_My_unclustered_transpose,U_record_sat_My_unclustered,L_record_sat_My_unclustered,best_L_sat_My_unclustered,num_candidate_sat_My_unclustered,time_sat_My_unclustered] =Sat_RotACM_My(data_3d_v,data_2d_N,branch_resolution ,epsilon_r,sample_resolution,verbose_flag,id);

    [min_error_sat_My_unclustered,R_opt_sat_My_unclustered]=min_error(num_candidate_sat_My_unclustered,R_sat_My_unclustered_transpose,R_gt);
    [alpha,phi,theta]=rot2angle(R_opt_sat_My_unclustered);
    found_inliers_sat_My_unclustered = abs(dot(R_opt_sat_My_unclustered'*data_3d_v',data_2d_N'))<=epsilon_r;
    %%%%%%  result record
    Record_SCM_FGO_unclustered(num+1,:)={num*10,min_error_sat_My_unclustered,0,0,time_sat_My_unclustered,num_candidate_sat_My_unclustered,...
                            length(data_3d_v),length(unique(id(gt_inliers))),...
                            length(unique(id(found_inliers_sat_My_unclustered))),best_L_sat_My_unclustered,semantic_mode_ratio,alpha,phi,theta};

    %%%%%%%%%%%%%%%%%%%%%%%  RAL_sat_clustered %%%%%%%%%%%%%%%%%%%%%
    % %%%%%%%%% rot
    % [R_sat_RAL_trans,U_record_sat_RAL,L_record_sat_RAL,best_L_sat_RAL,num_candidate_sat_RAL,time_sat_RAL] = Sat_RotACM_RAL(data_3d_v_clustered',data_2d_N_clustered', epsilon_r, branch_resolution,verbose_flag,id_cluster,kernel);
    [R_sat_RAL_trans,U_record_sat_RAL,L_record_sat_RAL,best_L_sat_RAL,num_candidate_sat_RAL,time_sat_RAL] = Sat_RotACM_RAL(data_3d_v_clustered',data_2d_N_clustered', epsilon_r, branch_resolution,verbose_flag,id_cluster);

    [min_error_sat_RAL,R_opt_sat_RAL]=min_error(num_candidate_sat_RAL,R_sat_RAL_trans,R_gt);
    [alpha,phi,theta]=rot2angle(R_opt_sat_RAL);
    found_inliers_sat_RAL = abs(dot(R_opt_sat_RAL'*data_3d_v_clustered',data_2d_N_clustered'))<=epsilon_r;
    %%%%%%%%% result record
    Record_SCM_EGO_clustered(num+1,:)={num*10,min_error_sat_RAL,0,0,time_sat_RAL,num_candidate_sat_RAL,...
                            length(data_3d_v_clustered),length(unique(id_cluster(gt_inliers))),...
                            length(unique(id_cluster(found_inliers_sat_RAL))),best_L_sat_RAL,semantic_mode_ratio,alpha,phi,theta};

    % %%%%%%%%%%%%%%%%%%%%%%%  My_unsat_unclustered %%%%%%%%%%%%%%%%%%%
    % %%%%%%%%% rot
    [R_plain_My_transpose,U_record_plain_My,L_record_plain_My,best_L_plain_My,num_candidate_plain_My,time_plain_My] =RotACM_My(data_3d_v,data_2d_N,branch_resolution ,epsilon_r,sample_resolution,verbose_flag);
    [min_error_plain_My,R_opt_plain_My]=min_error(num_candidate_plain_My,R_plain_My_transpose,R_gt);
    [alpha,phi,theta]=rot2angle(R_opt_plain_My);
    found_inliers_plain_My = abs(dot(R_opt_plain_My'*data_3d_v',data_2d_N'))<=epsilon_r;

    %%%%%%%% result record
    Record_CM_FGO(num+1,:)={num*10,min_error_plain_My,0,0,time_plain_My,num_candidate_plain_My,...
                            length(data_3d_v),nnz(lines2d(:,end)>=0),...
                            length(unique(id(found_inliers_plain_My))),best_L_plain_My,semantic_mode_ratio,alpha,phi,theta};
    % 
    % 
    % %%%%%%%%%%%%%%%%%%%%%%%  RAL_unsat_unclustered %%%%%%%%%%%%%%%%%%
    % %%%%%%%%% rot
    [R_plain_RAL,U_record_plain_RAL,L_record_plain_RAL,best_L_plain_RAL,num_candidate_plain_RAL,time_plain_RAL] =Rot_RAL(data_3d_v',data_2d_N',epsilon_r,branch_resolution,verbose_flag);
    [min_error_plain_RAL,R_opt_plain_RAL]=min_error(num_candidate_plain_RAL,R_plain_RAL,R_gt);
    [alpha,phi,theta]=rot2angle(R_opt_plain_RAL);
    found_inliers_plain_RAL = abs(dot(R_opt_plain_RAL'*data_3d_v',data_2d_N'))<=epsilon_r;
    %%%%%%%%% result record
    Record_CM_EGO(num+1,:)={num*10,min_error_plain_RAL,0,0,time_plain_RAL,num_candidate_plain_RAL,...
                            length(data_3d_v),nnz(lines2d(:,end)>=0),...
                            length(unique(id(found_inliers_plain_RAL))),best_L_plain_RAL,semantic_mode_ratio,alpha,phi,theta};
    

end
toc
%% 

Record_SCM_FGO_clustered(all(Record_SCM_FGO_clustered{:, :} == 0, 2), :) = [];
Record_SCM_EGO_clustered(all(Record_SCM_EGO_clustered{:, :} == 0, 2), :) = [];
Record_SCM_FGO_unclustered(all(Record_SCM_FGO_unclustered{:, :} == 0, 2), :) = [];
Record_CM_FGO(all(Record_CM_FGO{:, :} == 0, 2), :) = [];
Record_CM_EGO(all(Record_CM_EGO{:, :} == 0, 2), :) = [];

%%
function []=plot_bound_record(L_record,U_record)
    plot(1:length(U_record),U_record,'Color','b')
    hold on
    plot(1:length(L_record),L_record,'Color','b')
    hold off
end
%% 
% 
% Record_RAL_sat_clustered(all(Record_RAL_sat_clustered{:, :} == 0, 2), :) = [];
% rot_error = Record_RAL_sat_clustered{:,2};
% mean(rot_error>10)*100
%%
save(output_filename,"Record_CM_EGO","Record_CM_FGO","Record_SCM_FGO_unclustered","Record_SCM_EGO_clustered", ...
                            "Record_SCM_FGO_clustered","sampleSize","sample_resolution","epsilon_r","branch_resolution");
% % 

