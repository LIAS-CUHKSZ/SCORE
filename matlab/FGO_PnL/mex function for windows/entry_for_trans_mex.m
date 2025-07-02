%% entry for Sat_Bounds_trans
clear
clc
dataset_idx = "69e5939669";
space_size = [10.3, 6, 2.6];
data_folder="csv_dataset/"+dataset_idx+"/";
load(data_folder+"lines3D.mat");
%%% params
branch_reso = 0.03; % terminate bnb when branch size <= branch_reso
sample_reso = 0.03; % resolution for interval analysis
prox_thres  = 0.01; %
%%% rotation data
Record_SCM_FGO = load("matlab\Experiments\records\"+dataset_idx+"_rotation_record.mat").("Record_SCM_FGO");
epsilon_rs = Record_SCM_FGO.epsilon_r;
valid_idx = Record_SCM_FGO{:,1}; % These images have passed the rotation ambiguity test.
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
[pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,score_t_gt_CM,kernel_buff_SCM,score_t_gt_SCM] = ...
    under_specific_rot(num_2D_lines,ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r,t_gt,epsilon_t,intrinsic);
%%%
ceil_y = ceil(space_size(2)); ceil_z = ceil(space_size(3));
branch = zeros(6,ceil_y*ceil_z);  % each column: y_min, z_min, y_max, z_max, lower bound, upper bound
best_lower = -1; t_best=zeros(3,1);
for i=1:ceil_y
    for j=1:ceil_z
        idx = (i-1)*ceil_z+j;
        br_ = [i-1;j-1;i;j];
        [u_,l_,t_sample] = Sat_Bounds_trans(pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),id_inliers_under_rot,epsilon_t,br_,space_size,kernel_buff_SCM,prox_thres);
        branch(:,idx)=[br_;u_;l_];
        if l_ > best_lower
            t_best = t_sample;
            best_lower = l_;
        elseif l_== best_lower
            t_best = [t_best,t_sample];
        else
        end
    end
end

%% entry for trans_upper_interval
%%% generate boundary
sample_reso=0.01;
branch=[0;0;1;1];
if branch(3)-branch(1)>=sample_reso
    y_grid = branch(1):sample_reso:branch(3);
    z_grid = branch(2):sample_reso:branch(4);
    if y_grid(end)~=branch(3)
        y_grid=[y_grid,branch(3)];
        z_grid=[z_grid,branch(4)];
    end
else
    y_grid = [branch(1),branch(3)];
    z_grid = [branch(2),branch(4)];
end
M = length(y_grid); N = length(z_grid);
boundary = zeros(2,(M+N)*2-4);
boundary(:,1:N)=[y_grid(1)*ones(1,N);z_grid];
boundary(:,N:N+M-1)=[y_grid;z_grid(end)*ones(1,M)];
boundary(:,N+M-1:2*N+M-2)=[y_grid(end)*ones(1,N);z_grid(end:-1:1)];
boundary(:,2*N+M-2:(M+N)*2-4)=[y_grid(end:-1:2);z_grid(1)*ones(1,M-1)];
n_2D_rot = [cosd(30),sind(30),0];
p_3D = [3,4,5];
epsilon_t = 0.12;
x_limit = 5;
trans_upper_interval(n_2D_rot,p_3D,epsilon_t,x_limit,boundary)

%%
function [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,score_t_gt_CM,kernel_buff_SCM,score_t_gt_SCM] = ...
    under_specific_rot(num_2D_lines,ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r,t_gt,epsilon_t,intrinsic)

    inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
    id_inliers_under_rot = ids(inlier_under_rot);
    %%% saturation function design
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(id_inliers_under_rot==i);
    end
    kernel_buff_SCM = zeros(num_2D_lines,max(match_count));
    kernel_buff_CM  = ones(num_2D_lines,max(match_count));
    for i = 1:num_2D_lines
        kernel_buff_SCM(i,1)=1;
        for j = 2:match_count(i)
            kernel_buff_SCM(i,j)=log(j)-log(j-1);
        end
    end
    kernel_buff_SCM(:,2:end)=kernel_buff_SCM(:,2:end)/sum(sum(kernel_buff_SCM(:,2:end)));
    %%%
    n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
    endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
    %%% fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
    pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
    residuals = sum(pert_rot_n_2D_inlier.*(endpoints_3D_inlier(1:2:end,:)-t_gt'),2);
    inliers_t_gt = find(abs(residuals )<=epsilon_t);
    inliers_t_gt = prune_inliers(R_opt,intrinsic,inliers_t_gt,endpoints_3D_inlier,t_gt);
    score_t_gt_SCM = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_SCM);
    score_t_gt_CM = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_CM);
end
