clear;
clc;
% we use the data from one picture in scene c173f62b15
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dataload  %%%%%%%%%%%%%%%%%
load("matlab/test_demo/lines3d.mat");
lines2D = readmatrix("matlab/test_demo/test2dlines.csv");
T_gt = readmatrix("matlab/test_demo/gt_pose.csv");
K_p=readmatrix("matlab/test_demo/camera_intrinsic.csv");
test_room_size=[10.4 , 5 ,3.3]; % room size for c173f62b15
K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
lines2D(:,1:3)=lines2D(:,1:3)*K;
lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
%%%%%%%%%%%%%%   kernel functions %%%%%%%%%%%
kernel_SCM = @(x) x^-9;
trunc_num = 100; kernel_buffer_SCM=zeros(trunc_num,1);
for i=1:trunc_num
    kernel_buffer_SCM(i)=kernel_SCM(i);
end
kernel_buffer_CM = 1:1:trunc_num;
kernel_buffer_CM = kernel_buffer_CM';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  params  %%%%%%%%%%%%%%%%%
% rotation bnb
verbose_flag=0; mex_flag=1;
branch_reso = pi/512; sample_reso = pi/512;
prox_thres = 3/180*pi;
epsilon_r = 0.0085;
% translation ransac
sampleSize=4;
ransac_iterations=10000;
epsilon_t = 0.025;
%%  experiment
 [n_2D_cluster,v_3D_cluster,id_cluster,~]=match_line(lines2D,lines3D_cluster);
%%%%%%%%%%%%%%%%%%%%%%%%%%% clustered %%%%%%%%%%%%%%%%%%%%%%%%%%
gt_inliers_idx = abs(dot(R_gt'*v_3D_cluster',n_2D_cluster'))<=epsilon_r;
gt_inliers_id = id_cluster(gt_inliers_idx);
gt_score = calculate_score(gt_inliers_id,kernel_SCM);
num_2D_line_match=length(unique(gt_inliers_id));
% CM_FGO_clustered
[R_opt_top,best_score,num_candidate,time,~,~] = ...
    Sat_RotFGO(n_2D_cluster,v_3D_cluster,id_cluster,kernel_buffer_SCM,...
    branch_reso,epsilon_r,sample_reso,prox_thres,verbose_flag,mex_flag);
[min_err,R_opt]=min_error(num_candidate,R_opt_top,R_gt);


%%%%%% rot estimation %%%%%%%%%%%%%
%%%%%%%%%%%%% plain_version
[R_sat_My_transpose,U_record_sat_My,L_record_sat_My,best_L_sat_My,num_candidate_sat_My,time_sat_My] =Sat_RotFGO(data_3d_v_clustered,data_2d_N_clustered,branch_reso ,epsilon_r,sample_reso,verbose_flag,id_cluster,kernel);
R_opt_sat_My = R_sat_My_transpose';
found_inliers_sat_My = abs(dot(R_opt_sat_My'*data_3d_v_clustered',data_2d_N_clustered'))<=epsilon_r;
index_inlier=find(abs(dot(R_opt_sat_My'*data_3d_v',data_2d_N'))<=epsilon_r);
id_inliers = id(index_inlier);
N_2d_rot_inlier=data_2d_N(index_inlier,:);
c_3d_rot_inlier=data_3d_c(index_inlier,:);
v_3d_rot_inlier=data_3d_v(index_inlier,:);

%%%%%%   translation estimation  %%%%%%%%%%%%%
if length(unique(id_inliers))<4
    sprintf("not enough inliers")
    return
else
    [t_opt_sat_var,num_candidate,max_score] = Sat_t_ransac(N_2d_rot_inlier',c_3d_rot_inlier',R_opt_sat_My,id_inliers,sampleSize,ransac_iterations,epsilon_t,kernel,test_room_size);
end

%% Error calculation
fprintf('===== Error Analysis =====\n');
% Rotation error  
R_error = angular_distance(R_opt_sat_My, R_gt);
% translation error  
t_error = norm(t_opt_sat_var - t_gt);
%%%  because the high outlier ratio, there maybe more than one solution
fprintf('Rotation Error: %.4f degrees\n', R_error);
fprintf('Translation Error: %.4f units\n', t_error);



%%  visualization of the upper and lower bounds
function []=plot_bound_record(L_record,U_record)
    plot(1:length(U_record),U_record,'Color','b')
    hold on
    plot(1:length(L_record),L_record,'Color','b')
    hold off
end
