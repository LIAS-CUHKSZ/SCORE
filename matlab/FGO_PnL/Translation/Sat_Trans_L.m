%%%%
% calculate lower bound for Sat-Translation problem in PnL

%%% Inputs:
% pert_rot_n : N x 3,  perturbed and rotated normal vector for each 2D line
% p_3D       : N x 3,  points on 3D lines
% ids        : N x 1,  2D line idx for each association
% epsilon_t  : scalar, error tolerance
% br_        : 4 x 1,  sub-cube
% space_size : 3 x 1,  xyz range of the scene
% sat_buff   : M x N, store weights given by the selected saturation function.
% prox_thres : scalar, used for clustering proximate stabbers

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [lower_bound,x_opt] = Sat_Trans_L(pert_rot_n,p_3D,ids,epsilon_t,br_,space_size,sat_buffer,prox_thres)
    
    N = size(pert_rot_n,1);
    x_opt = [];
    % --- 0. constrain the sub cube within the scene ---
    x_limit = space_size(1);
    br_(3)=min(br_(3),space_size(2));
    br_(4)=min(br_(4),space_size(3));
    if br_(3)<=br_(1) || br_(4)<=br_(2)
        lower_bound=-1;
        return
    end

    % --- 1. prepare the intervals for stabbing ---
    % sample the center point
    yz_sampled = [br_(1)+br_(3);br_(2)+br_(4)]/2; 
    intervals_lower = []; ids_lower=[];
    for i = 1:N
        [tmp_interval] = trans_lower_interval(pert_rot_n(i,:),p_3D(i,:),epsilon_t,yz_sampled,x_limit);
        intervals_lower=[intervals_lower;tmp_interval];
        ids_lower = [ids_lower;ids(i)*ones(length(tmp_interval)/2,1)];
    end

    % --- 2. saturated interval stabbing ---
    if isempty(ids_lower) % no valid interval
        lower_bound = -1; t_sample = [];
    else
        [lower_bound, x_opt] = saturated_interval_stabbing(intervals_lower,ids_lower,sat_buffer,prox_thres);
    end
end
