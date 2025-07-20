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
load(data_folder+"lines3D.mat");
%%% rotation bnb
verbose_flag=0; % verbose mode for BnB
mex_flag=1; % use matlab mex code for acceleration
branch_resolution = pi/512; % terminate bnb when branch size <= branch_reso
sample_resolution = pi/512; % resolution for interval analysis
% paramaters for handling unbiguity of the global optimum
% basically we keep all the candidates which
% (a) have the same score after rounding (b) not proximate to each other
round_digit = 9;
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
%%% check ambiguity
M = size(lines2D,1);
n_2D_gt = zeros(M,3);
v_3D_gt = zeros(M,3);
residual_r = zeros(M,1);
for i=1:M
    matched_idx = int32(lines2D(i,9))+1;
    n = lines2D(i,1:3);
    n_2D_gt(i,:)=n;
    v = lines3D(matched_idx,4:6)-lines3D(matched_idx,1:3);
    v = v /norm(v);
    v_3D_gt(i,:)=v;
    residual_r(i)=abs((R_gt*n')'*v');
end
ids = 1:M;
ids = ids';
epsilon = max(residual_r)*1.2;

%%%
line_pair_data = data_process(n_2D_gt,v_3D_gt);
branch=[];
Branch=[0;0;pi;pi]; 
kernel_buff=zeros(M,10);
kernel_buff(:,1)=1;
Sat_Bounds_FGO(line_pair_data,Branch,epsilon,sample_resolution,ids,kernel_buff,prox_thres);  



% N = line_pair.size;
% u_center = polar_2_xyz(0.5*(Branch(1)+Branch(3)),0.5*(Branch(2)+Branch(4)));
% h1_center = zeros(N,1);
% h2_center = zeros(N,1);
% for i = 1:N
%     h1_center(i) = dot(u_center,line_pair.outer_product(i,:));
%     h2_center(i) = dot(u_center,line_pair.vector_n(i,:))*dot(u_center,line_pair.vector_v(i,:))-line_pair.inner_product(i);
% end
% [A_center,phi_center,const_center] = cal_params(line_pair.inner_product,h1_center,h2_center);
% intervals_lower = []; ids_lower=[];
% for i = 1:N
%     [tmp_interval] = lower_interval(A_center(i),phi_center(i),const_center(i),epsilon);
%     intervals_lower=[intervals_lower;tmp_interval];
%     ids_lower = [ids_lower;id(i)*ones(length(tmp_interval)/2,1)];
% end
% [Q_lower, theta_lower] = saturated_interval_stabbing(intervals_lower,ids_lower,kernel_buffer,prox_thres);
% [h1_upper,h1_lower] = h1_interval_mapping(line_pair,Branch,sample_resolution);
% [h2_upper,h2_lower] = h2_interval_mapping(line_pair,Branch,sample_resolution);
% 
% %%%
% function [A,phi,const] = cal_params(product, h1 ,h2)
%     %%% f =  product + sin(theta)* h1 + (1-cos theta )* h2  
%     %%%   =  h1 sin(theta) - h2 cos(theta) + product + h2
%     A = sqrt(h1.^2 + h2.^2);
%     phi = atan2(-h2,h1);
%     phi =phi.*(phi>=0) + (phi+2*pi).*(phi<0); % make sure that phi in [0,2*pi]
%     const = product+h2;
% end


