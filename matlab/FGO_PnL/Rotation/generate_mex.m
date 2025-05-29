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
%%%  params
trunc_num=100;
kernel_buffer=zeros(trunc_num,1);
kernel_sat = @(x) x^-8;
for i=1:trunc_num
    kernel_buffer(i)=kernel_sat(i);
end
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
img_idx=0;
%%% read 2D line data of cur image
frame_id = sprintf("%06d",img_idx);
% lines2D(Nx9): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1)
lines2D = readmatrix(data_folder+"lines2d\frame_"+frame_id+"2dlines.csv");
lines2D = lines2D(lines2D(:,4)~=0,:); % delete 2D line without a semantic label
K_p=readmatrix(data_folder+"intrinsics\frame_"+frame_id+".csv");
K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
T_gt = readmatrix(data_folder+"poses\frame_"+frame_id+".csv");
R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
lines2D(:,1:3)=lines2D(:,1:3)*K;
lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
%%% check ambiguity
M = length(lines2D);
n_2D_gt = zeros(M,3);
v_3D_gt = zeros(M,3);
residual_r = zeros(M,1);
for i=1:M
    matched_idx = int32(lines2D(i,end))+1;
    n = lines2D(i,1:3);
    n_2D_gt(i,:)=n;
    v = lines3D(matched_idx,4:6);
    v_3D_gt(i,:)=v;
    residual_r(i)=abs((R_gt*n')'*v');
end
id_gt = 1:M;
id_gt = id_gt';
epsilon_r = max(residual_r)*1.2;
%%%
line_pair_data = data_process(n_2D_gt,v_3D_gt);
branch=[];
B_east=[0;0;pi;pi]; B_west=[0;pi;pi;2*pi];
[upper_east,lower_east,theta_east]=Sat_Bounds_FGO(line_pair_data,B_east,epsilon_r,sample_reso,id_gt,kernel_buffer);    


