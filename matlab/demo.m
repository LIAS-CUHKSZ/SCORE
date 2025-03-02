clear;
clc;
%%%%%%  tips %%%%%%%%%%%%%%
% The data used by this demo is randomly generated, if you want to have a 
% fast run, you can use mex_version(line 100).  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% dataload  %%%%%%%%%%%%%%%%%
load("test_data/lines3d.mat");
lines2d = readmatrix("test_data\test2dlines.csv");
T_gt = readmatrix("test_data\gt_pose.csv");
K_p=readmatrix("test_data\camera_intrinsic.csv");
if length(lines2d)<21
    sprintf("not enough 2d lines!!")
    return
end
test_room_size=[10.3,6,2.6];  %  you should change this when you use other dataset
semantic_label_mode = mode(lines2d(:,4));
semantic_mode_ratio = sum(lines2d(:,4)==mode(lines2d(:,4)))/length(lines2d);
K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
lines2d(:,1:3)=lines2d(:,1:3)*K;
lines2d(:,1:3)=lines2d(:,1:3)./vecnorm(lines2d(:,1:3)')';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%   satured kernel (mex_version default kernel_3) %%%%%%%%%%%
kernel_1 = @(x) exp(-x+1);
kernel_2 = @(x) x^-2;
kernel_3 = @(x) x^-8;
kernel_4 = @(x) 2^(-x+1)/x;
kernel_5 = @(x) 1/factorial(x);
kernel_6 = @(x) 2^(-x+1);
kernel_7 = @(x) 2/x/(x+1);
kernel_8 = @(x) 1-(x>1);
kernel = kernel_3;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  params  %%%%%%%%%%%%%%%%%
% rotation bnb
verbose_flag=0;
epsilon_r = 0.01;
branch_resolution = pi/512;
sample_resolution = pi/16;
% translation ransac
sampleSize=4;
ransac_iterations=10000;
epsilon_t = 0.025;

%%  experiment
total_match_num=0;
for i=1:size(lines2d,1)
    idx_matched_3d = find(abs(lines3d(:,7)-lines2d(i,4))<0.1);
    total_match_num = total_match_num+length(idx_matched_3d);
end
data_3d_v=zeros(total_match_num,3);
data_2d_N=zeros(total_match_num,3);
data_3d_c=zeros(total_match_num,3);
id=zeros(total_match_num,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%% clustered %%%%%%%%%%%%%%%%%%%%%%%%%%
clustered_match_num=0;
for i=1:size(lines2d,1)
    idx_matched_3d = find(abs(lines3d_cluster(:,7)-lines2d(i,4))<0.1);
    clustered_match_num = clustered_match_num+length(idx_matched_3d);
end
data_3d_v_clustered=zeros(clustered_match_num,3);
data_2d_N_clustered=zeros(clustered_match_num,3);
id_cluster=zeros(clustered_match_num,1);
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


%%%%%% rot estimation %%%%%%%%%%%%%

%%%%%%%%%%%%% plain_version
% [R_sat_My_transpose,U_record_sat_My,L_record_sat_My,best_L_sat_My,num_candidate_sat_My,time_sat_My] =Sat_RotACM_My(data_3d_v_clustered,data_2d_N_clustered,branch_resolution ,epsilon_r,sample_resolution,verbose_flag,id_cluster,kernel);
%%%%%%%%%%%%% mex_version
[R_sat_My_transpose,U_record_sat_My,L_record_sat_My,best_L_sat_My,num_candidate_sat_My,time_sat_My] =Sat_RotACM_My(data_3d_v_clustered,data_2d_N_clustered,branch_resolution ,epsilon_r,sample_resolution,verbose_flag,id_cluster);

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
    [t_opt_sat_var,num_candidate,max_score] = sat_var_t_Ransac(N_2d_rot_inlier',c_3d_rot_inlier',R_opt_sat_My,id_inliers,sampleSize,ransac_iterations,epsilon_t,kernel,test_room_size);
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
