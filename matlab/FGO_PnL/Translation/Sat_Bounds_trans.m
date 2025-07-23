%%%%
% calculate lower and upper bound for a given sub-cube.

%%% Inputs:
% pert_rot_n : N x 3,  perturbed and rotated normal vector for each 2D line
% p_3D       : N x 3,  points on 3D lines
% ids        : N x 1,  2D line idx for each association
% R_opt      : 3 x 3,  upstream optimal rotation estimate
% epsilon_t  : scalar, error tolerance
% br_        : 4 x 1,  sub-cube
% space_size : 3 x 1,  xyz range of the scene
% prox_thres : scalar, for merging stabber in interval stabbing

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [upper_bound,lower_bound,t_sample] = Sat_Bounds_trans(pert_rot_n,p_3D,ids,epsilon_t,br_,space_size,kernel_buffer,prox_thres)
    
    N = size(pert_rot_n,1);
    
    % constrain the sub cube within the scene
    x_limit = space_size(1);
    br_(3)=min(br_(3),space_size(2));
    br_(4)=min(br_(4),space_size(3));
    if br_(3)<=br_(1) || br_(4)<=br_(2)
        upper_bound=-1;
        lower_bound=-1;
        t_sample=[];
        return
    end
    vertices = [br_(1),br_(2); br_(1),br_(4);
                br_(3),br_(2); br_(3),br_(4)];

    % --- 1. upper_bound ---
    % prepare the intervals for stabbing
    intervals_upper = []; ids_upper=[];
    for i=1:N
        [tmp_interval] = trans_upper_interval(pert_rot_n(i,:),p_3D(i,:),epsilon_t,x_limit,vertices);
        intervals_upper=[intervals_upper;tmp_interval];
        ids_upper = [ids_upper;ids(i)*ones(size(tmp_interval,1)/2,1)];
    end
    if isempty(ids_upper) % no valid interval
        upper_bound = -1; lower_bound = -1; t_sample = [];
        return 
    else
        [upper_bound,~] = saturated_interval_stabbing(intervals_upper,ids_upper,kernel_buffer,prox_thres);
    end

    % --- 2. upper_bound ---
    yz_sampled = [br_(1)+br_(3);br_(2)+br_(4)]/2; % sample the center point
    % prepare the intervals for stabbing
    intervals_lower = []; ids_lower=[];
    for i = 1:N
        [tmp_interval] = trans_lower_interval(pert_rot_n(i,:),p_3D(i,:),epsilon_t,yz_sampled,x_limit);
        intervals_lower=[intervals_lower;tmp_interval];
        ids_lower = [ids_lower;ids(i)*ones(length(tmp_interval)/2,1)];
    end
    if isempty(ids_lower) % no valid interval
        lower_bound = -1; t_sample = [];
    else
        [lower_bound, x_opt] = saturated_interval_stabbing(intervals_lower,ids_lower,kernel_buffer,prox_thres);
        num_stabber = size(x_opt,2);
        t_sample = zeros(3,num_stabber);
        t_sample(2:3,:)=repmat(yz_sampled,[1,num_stabber]);
        t_sample(1,:)=x_opt';
    end
end
