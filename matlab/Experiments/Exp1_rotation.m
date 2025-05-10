%%%%
% Orientation Estimation
% FGO vs EGO
% Saturated Consensus Maximization vs Consensus Maximization

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

clear;
clc;
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(2);
data_folder="csv_dataset/"+dataset_idx+"/";
load(data_folder+"lines3D.mat");

%%% statistics
total_img=1000;
column_names=...
    ["image id","time","orient err","# 2D lines with match","score","score under gt","# candidates"];
columnTypes =...
    ["int32","double","double","int32","double","double","int32"];
Record_SCM_FGO_clustered     =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
%
column_names2=...
    ["image id","orient err","score","score under gt","gt_x","gt_y","gt_z","pert_x","pert_y","pert_z"];
columnTypes2 =...
    ["int32","double","double","double","double","double","double","double","double","double"];
Largerr_SCM_FGO_clustered     =table('Size', [total_img, length(column_names2)],'VariableTypes', columnTypes2,'VariableNames', column_names2);
%%
%%%  params
kernel = @(x) x^-8;
trunc_num=100;
kernel_buffer=zeros(trunc_num,1);
for i=1:trunc_num
    kernel_buffer(i)=kernel(i);
end
line_num_thres=15; % minimal number of 2D lines required in the image
%%% rotation bnb
verbose_flag=0; % verbose mode for BnB
mex_flag=1; % use matlab mex code for acceleration
branch_reso = pi/512; % terminate bnb when branch size <= branch_reso
sample_reso = pi/512; % resolution for interval analysis
% paramaters for handling unbiguity of the global optimum
% basically we keep all the candidates which 
% (a) have the same score after rounding (b) not proximate to each other
round_digit = 9;
prox_thres = cosd(5);
for num =70:total_img
    img_idx=num*10;
    %%% read 2D line data of cur image
    frame_id = sprintf("%06d",img_idx);
    if ~exist(data_folder+"lines2d\frame_"+frame_id+"2dlines.csv",'file')
        continue
    end
    img_idx
    % lines2D(Nx10): normal vector(3x1), semantic label(1), projection error(orient,trans), endpoint a(u,v), endpoint b(u,v) 
    lines2D = readmatrix(data_folder+"lines2d\frame_"+frame_id+"2dlines.csv"); 
    lines2D = lines2D(lines2D(:,4)~=0,:); % delete 2D line without a semantic label
    if length(lines2D)<line_num_thres
        continue
    end
    K_p=readmatrix(data_folder+"intrinsics\frame_"+frame_id+".csv");
    K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(data_folder+"poses\frame_"+frame_id+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    lines2D(:,1:3)=lines2D(:,1:3)*K; 
    lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    %%% match 2D and 3D lines using semnatic label
    % match with unclustered 3D lines 
    [n_2D,v_3D,id,~]=match_line(lines2D,lines3D); 
    % fprintf("# match with all 3D lines:       %d\n",length(id));
    
    % match with clustered 3D lines 
    [n_2D_cluster,v_3D_cluster,id_cluster]=match_line(lines2D,lines3D_cluster);
    % fprintf("# match with clustered 3D lines: %d\n",length(id_cluster)); 

    %%%%%%%%%%%%%%%%%%% Estimate Orientation %%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % set threshold
    epsilon_r=max(lines2D(:,6))*1.05;
    %
    gt_inliers_idx = find(abs(dot(R_gt'*v_3D_cluster',n_2D_cluster'))<=epsilon_r);

    gt_inliers_id = id_cluster(gt_inliers_idx);
    gt_score = calculate_score(gt_inliers_id,kernel_buffer);
    num_2D_line_match=length(unique(gt_inliers_id));
    % Sat_FGO_clustered
    [R_opt_top,best_score,num_candidate,time,~,~] = ...
        Sat_RotFGO(n_2D_cluster,v_3D_cluster,id_cluster,kernel_buffer,...
        branch_reso,epsilon_r,sample_reso,round_digit,prox_thres,verbose_flag,mex_flag);
    [min_err,R_opt]=min_error(num_candidate,R_opt_top,R_gt);
    est_inliers_idx=find(abs(dot(R_opt'*v_3D_cluster',n_2D_cluster'))<=epsilon_r);
    est_inliers_id = id_cluster(est_inliers_idx);
    Record_SCM_FGO_clustered(num+1,:)={img_idx,time,min_err,num_2D_line_match,best_score,gt_score,num_candidate};
    if min_err > 160
        Delta_R=R_opt*R_gt';
        gt_rotv = rotmat2vec3d(R_gt);
        err_rotv = rotmat2vec3d(Delta_R);
        Largerr_SCM_FGO_clustered(num+1,:)={img_idx,min_err,best_score,gt_score,...
            gt_rotv(1),gt_rotv(2),gt_rotv(3),err_rotv(1),err_rotv(2),err_rotv(3)};
    end
end
Largerr_SCM_FGO_clustered(Largerr_SCM_FGO_clustered.("score")==0,:)=[];
Record_SCM_FGO_clustered(Record_SCM_FGO_clustered.("score")==0,:)=[];
output_filename= "./matlab/Experiments/records/"+dataset_idx+"_rotation_record.mat";
save(output_filename,"Record_SCM_FGO_clustered","Largerr_SCM_FGO_clustered");
%%

N_large_err=height(Largerr_SCM_FGO_clustered);
sz=5; scale=1;
for n=1:N_large_err
    fprintf("image idx:%d, max score:%f, score under gt:%f",Largerr_SCM_FGO_clustered.("image id")(n),Largerr_SCM_FGO_clustered.("score")(n),Largerr_SCM_FGO_clustered.("score under gt")(n));
    fprintf("\n");
    % plot the gt camera orientation
    rot_gt_v = Largerr_SCM_FGO_clustered{n,5:7};
    R_gt = rotvec2mat3d(rot_gt_v);
    T_gt = [R_gt,zeros(3,1);0,0,0,1];
    draw_pose(T_gt,sz,scale);
    hold on
    % plot the estimated camera orientation
    rot_pert_v = Largerr_SCM_FGO_clustered{n,8:10};
    R_pert = rotvec2mat3d(rot_pert_v);
    R_est = R_pert*R_gt;
    T_est = [R_est,ones(3,1);0,0,0,1];
    draw_pose(T_est,sz,scale);
    hold off
end
















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

