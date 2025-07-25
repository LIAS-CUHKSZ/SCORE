%% entry for Sat_Bounds_trans
clear
clc
dataset_idx = "a1d9da703c";
space_size = [8,    6, 4];
data_folder="csv_dataset/"+dataset_idx+"/";
lines3D = readmatrix(data_folder+"3Dlines.csv");
%%% params
branch_reso = 0.03; % terminate bnb when branch size <= branch_reso
sample_reso = 0.03; % resolution for interval analysis
prox_thres  = 0.01; %
%%% rotation data
rotation_data = load("matlab\Experiments\records\gt_semantics\"+dataset_idx+"_rotation_record.mat");
epsilon_rs = 0.015;
valid_idx = rotation_data.Record_CM_FGO{:,1};
% ---------------------------------------------------------------------
% --- 1. load data ---
num = 1;
img_idx=valid_idx(num);
frame_id = sprintf("%06d",img_idx);
K_p=readmatrix(data_folder+"intrinsics\frame_"+frame_id+".csv");
intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
T_gt = readmatrix(data_folder+"poses\frame_"+frame_id+".csv"); R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/frame_"+frame_id+".csv")+1;
lines2D = readmatrix(data_folder+"lines2D\frame_"+frame_id+"_2Dlines.csv");
lines2D = lines2D(lines2D(:,4)~=0,:);   % delete 2D line without a semantic label
lines2D(:,1:3)=lines2D(:,1:3)*intrinsic;  lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
num_2D_lines = size(lines2D,1);
lines3D_sub = lines3D(retrived_3D_line_idx,:); % retrived sub-map
[ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D_sub);
epsilon_r = epsilon_rs(num);
epsilon_t = max(lines2D(:,11))+0.001;
R_opt = R_gt;
[pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
    preprocess_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r);
match_count = zeros(num_2D_lines,1);
for i = 1:num_2D_lines
    match_count(i) = sum(id_inliers_under_rot==i);
end
kernel_buff_CM = zeros(num_2D_lines,max(match_count));
kernel_buff_CM(:,1)=1;
%%%
br_ = [0;0;1;1];
p_3Ds = endpoints_3D_inlier(1:2:end,:);
[upper_bound] = Sat_Trans_U(pert_rot_n_2D_inlier,p_3Ds,id_inliers_under_rot,epsilon_t,br_,space_size,kernel_buff_CM);
[lower_bound,t_sample] = Sat_Trans_L(pert_rot_n_2D_inlier,p_3Ds,id_inliers_under_rot,epsilon_t,br_,space_size,kernel_buff_CM,prox_thres);
%%
function [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
    preprocess_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r)

    % find inliers satisfying the rotation constraint
    inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
    id_inliers_under_rot = ids(inlier_under_rot);
    n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
    endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
    
    % fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
    pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
end