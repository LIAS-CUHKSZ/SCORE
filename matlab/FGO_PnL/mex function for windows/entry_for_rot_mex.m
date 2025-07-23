%%%%
% Orientation Estimation
% FGO vs EGO
% Saturated Consensus Maximization vs Consensus Maximization

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
 
%%% License: MIT
%%%%

clear;
clc;
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(1);
data_folder="csv_dataset/"+dataset_idx+"/";
lines3D = readmatrix(data_folder+"3Dlines.csv");
%%% rotation bnb
branch_resolution = pi/512; % terminate bnb when branch size <= branch_reso
sample_resolution = pi/512; % resolution for interval analysis
prox_thres = cosd(5);
img_idx=70;
%%% read 2D line data of cur image
frame_id = sprintf("%06d",img_idx);
% lines2D(Nx9): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1)
lines2D = readmatrix(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv");
lines2D = lines2D(lines2D(:,4)~=0,:); % delete 2D line without a semantic label
K_p=readmatrix(data_folder+"intrinsics/frame_"+frame_id+".csv");
K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
T_gt = readmatrix(data_folder+"poses/frame_"+frame_id+".csv");
R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
lines2D(:,1:3)=lines2D(:,1:3)*K;
lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
lines2D(lines2D(:,9)==-1,:)=[];
M = size(lines2D,1);
% pre-process
n_2D_gt = zeros(M,3);
v_3D_gt = zeros(M,3);
for i=1:M
    matched_idx = int32(lines2D(i,9))+1;
    n = lines2D(i,1:3);
    n_2D_gt(i,:)=n;
    v = lines3D(matched_idx,4:6)-lines3D(matched_idx,1:3);
    v = v /norm(v);
    v_3D_gt(i,:)=v;
end
ids = 1:M; ids = ids';
line_pair_data = data_process(n_2D_gt,v_3D_gt);
kernel_buff=zeros(M,10); kernel_buff(:,1)=1;

% bound function to be compiled as mex
epsilon_r = 0.03;
Branch=[0;0;pi;pi]; 
[lower_bound,theta_opt] = Sat_Rot_L(line_pair_data,Branch,epsilon_r,ids,kernel_buff,prox_thres);
[upper_bound] = Sat_Rot_U(line_pair_data,Branch,epsilon_r,sample_resolution,ids,kernel_buff);
