% pert_rot_n : N x 3
% p_3D       : N x 3
% ids        : N x 1
% R_opt      : 3 x 3
% epsilon_t  : scalar
% br_        : 4 x 1
% sample_reso: scalar
% space_size : 3 x 1

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%
function [upper_bound,lower_bound,t_sample] = Sat_Bounds_trans(pert_rot_n,p_3D,ids,epsilon_t,br_,sample_reso,space_size,kernel_buffer)
    x_limit = space_size(1);
    br_(3)=min(br_(3),space_size(2));
    br_(4)=min(br_(4),space_size(3));
    if br_(3)<=br_(1) || br_(4)<=br_(2)
        upper_bound=-1;
        lower_bound=-1;
        t_sample=[];
        return
    end
    boundary = tranverse_boundary(br_,sample_reso);
    %%% upper_bound
    intervals_upper = []; ids_upper=[];
    N = size(pert_rot_n,1);
    for i=1:N
        [tmp_interval] = trans_upper_interval_mex(pert_rot_n(i,:),p_3D(i,:),epsilon_t,x_limit,boundary);
        intervals_upper=[intervals_upper;tmp_interval];
        ids_upper = [ids_upper;ids(i)*ones(size(tmp_interval,1)/2,1)];
    end
    if isempty(ids_upper)
        upper_bound = -1;
        lower_bound = -1;
        t_sample = [];
        return 
    else
        [upper_bound,~] = saturated_interval_stabbing(intervals_upper,ids_upper,kernel_buffer);
    end
    %%% lower bound
    yz_sampled = [br_(1)+br_(3);br_(2)+br_(4)]/2; % sample the center point
    intervals_lower = []; ids_lower=[];
    for i = 1:N
        [tmp_interval] = trans_lower_interval(pert_rot_n(i,:),p_3D(i,:),epsilon_t,yz_sampled,x_limit);
        intervals_lower=[intervals_lower;tmp_interval];
        ids_lower = [ids_lower;ids(i)*ones(length(tmp_interval)/2,1)];
    end
    if isempty(ids_lower)
        lower_bound = -1;
        t_sample = [];
    else
        [lower_bound, x_opt] = saturated_interval_stabbing(intervals_lower,ids_lower,kernel_buffer);
        x_opt = cluster_stabber(x_opt,sample_reso);
        num_stabber = size(x_opt,2);
        t_sample = zeros(3,num_stabber);
        t_sample(2:3,:)=repmat(yz_sampled,[1,num_stabber]);
        t_sample(1,:)=x_opt';
    end
end


function boundary = tranverse_boundary(branch,sample_reso)
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
end