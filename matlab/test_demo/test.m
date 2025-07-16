%%%%
% test demo of one query image
% for simplification, we remove image retriveal and code for selecting
% between multiple rotation candidates (which can be found in Experiments/d_rot_and_trans.m)
% Saturated Consensus Maximization vs Consensus Maximization
% GT rotation vs FGO_PnL rotation estimates

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT

% Note!! 
% If you don't want to or can't use the compiled mex functions, 
% remeber to set variables 'mex_flag=0' in functions Sat_RotFGO and Sat_TransFGO

clear
clc
space_size =  [10.5, 6, 3.0]; % for scene "69e5939669"
data_folder="matlab/test_demo/";
%%% rot params
prox_thres_r = 1*pi/180; % for clustering proximate stabbers
branch_reso_r = pi/256; % terminate bnb when branch size < branch_reso
sample_reso_r = pi/256; % resolution for interval analysis
epsilon_r = 0.015;
%%% trans params
branch_reso_t = 0.01; % terminate bnb when branch size <= branch_reso
prox_thres_t  = 0.01; %
epsilon_t = 0.03;
% ---------------------------------------------------------------------
% --- 1. load data ---
load(data_folder+"lines3D.mat");
K_p=readmatrix(data_folder+"camera_intrinsic.csv");
T_gt = readmatrix(data_folder+"gt_pose.csv");
R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
% this flag is used to initialize the BnB process
% set it at 2 when no prior information is provided
west_east_flag = 2;
intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1]; % intrinsic matrix
% lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1)
lines2D = readmatrix(data_folder+"2Dlines.csv");
lines2D(:,1:3)=lines2D(:,1:3)*intrinsic; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
% --- 2. semantic matching and saturation function design ---
[ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D);  % match with clustered 3D lines
num_2D_lines = size(lines2D,1);
match_count = zeros(num_2D_lines,1);
for i = 1:num_2D_lines
    match_count(i) = sum(ids==i);
end
rot_kernel_buff_CM = ones(num_2D_lines,max(match_count));    % classic CM
L = sum(log(match_count(match_count>0)));
rot_kernel_buff_SCM_entropy = zeros(num_2D_lines,max(match_count));
for i = 1:num_2D_lines
    if match_count(i)==0
        continue
    end
    rot_kernel_buff_SCM_entropy(i,1)=1-log(match_count(i))/L;
    for j = 2:match_count(i)
        rot_kernel_buff_SCM_entropy(i,j)=(log(j)-log(j-1))/L;
    end
end
%
outlier_ratio = 1-sum(match_count>0)/size(n_2D,1);
fprintf("%d 2D lines with %d associations at a outlier ratio of %f\n",num_2D_lines,length(ids),outlier_ratio)
%-------------------------------------------------------------
%---- 3. complete pipeline starts here -----
for o = 1:2
    % --- rotation estmation --- %
    if o==1
        fprintf("===relocalization with classic consensus maximization===\n")
        rot_kernel_buff = rot_kernel_buff_CM;
    else
        fprintf("===relocalization with saturated consensus maximization===\n")
        rot_kernel_buff = rot_kernel_buff_SCM_entropy;
    end
    time_all = 0;
    [R_opt_top,best_score,num_candidate_rot,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,rot_kernel_buff,...
        branch_reso_r,epsilon_r,sample_reso_r,prox_thres_r,west_east_flag);
    time_all = time_all+time;

    % --- translation estmation --- %
    R_opt = R_opt_top';
    [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
        under_specific_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r);
    %%% saturation function design
    match_count_pruned = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count_pruned(i) = sum(id_inliers_under_rot==i);
    end
    LL = sum(log(match_count_pruned(match_count_pruned>0)));
    trans_kernel_buff_SCM_entropy= zeros(num_2D_lines,max(match_count_pruned));
    trans_kernel_buff_CM = ones(num_2D_lines,max(match_count_pruned));
    for i = 1:num_2D_lines
        if match_count_pruned(i)==0
            continue
        end
        trans_kernel_buff_SCM_entropy(i,1)=1-log(match_count_pruned(i))/LL;
        for j = 2:match_count_pruned(i)
            trans_kernel_buff_SCM_entropy(i,j)=(log(j)-log(j-1))/LL;
        end
    end
    if o==1
        trans_kernel_buff = trans_kernel_buff_CM;
    else
        trans_kernel_buff=trans_kernel_buff_SCM_entropy;
    end
    [t_best_candidates,~,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,trans_kernel_buff,space_size,branch_reso_t,epsilon_t,prox_thres_t);
    time_all = time_all+time;
    % prune candidates according to geometric constraints
    [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,trans_kernel_buff);
    t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
    rot_err = angular_distance(best_R,R_gt);
    t_err   = norm(best_t-t_gt);
    fprintf("rot err:%f, trans err:%f, time: %f\n",rot_err,t_err,time_all);
end
%%
% ---------------------------------------------------------------------
% --- sub-functions ---
function [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
    under_specific_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r)

inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
id_inliers_under_rot = ids(inlier_under_rot);
n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
%%% fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
end

function [best_score,t_real_candidate] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D,endpoints_3D,ids,epsilon_t,t_candidates,kernel_buff)
p_3D = endpoints_3D(1:2:end,:);
best_score = -1; t_real_candidate=[];
for k=1:size(t_candidates,2)
    t_test = t_candidates(:,k);
    residuals = sum(pert_rot_n_2D.*(p_3D-t_test'),2);
    inliers = find(abs(residuals )<=epsilon_t);
    %%% the above inliers satisfy the geometric constraints,
    %%% we urther filter lines behind the camera and outside image
    real_inliers = prune_inliers(R_opt,intrinsic,inliers,endpoints_3D,t_test);
    score=calculate_score(ids(real_inliers),kernel_buff);
    if score > best_score
        best_score = score; t_real_candidate = t_test;
    elseif score == best_score
        t_real_candidate = [t_real_candidate,t_test];
    else
    end
end
end
