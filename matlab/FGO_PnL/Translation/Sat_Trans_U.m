%%%%
% calculate upper bound for Sat-Translation problem in PnL

%%% Inputs:
% pert_rot_n : N x 3,  perturbed and rotated normal vector for each 2D line
% p_3D       : N x 3,  points on 3D lines
% ids        : N x 1,  2D line idx for each association
% epsilon_t  : scalar, error tolerance
% br_        : 4 x 1,  sub-cube
% space_size : 3 x 1,  xyz range of the scene
% sat_buff   : M x N, store weights given by the selected saturation function.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [upper_bound] = Sat_Trans_U(pert_rot_n,p_3D,ids,epsilon_t,br_,space_size,sat_buffer)
N = size(pert_rot_n,1);
    
    % --- 0. constrain the sub cube within the scene ---
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

    % --- 1. prepare the intervals for stabbing ---
    intervals_upper = []; ids_upper=[];
    for i=1:N
        [tmp_interval] = trans_upper_interval(pert_rot_n(i,:),p_3D(i,:),epsilon_t,x_limit,vertices);
        intervals_upper=[intervals_upper;tmp_interval];
        ids_upper = [ids_upper;ids(i)*ones(size(tmp_interval,1)/2,1)];
    end

    % --- 2. saturated interval stabbing ---
    if isempty(ids_upper) % no valid interval
        upper_bound = -1; lower_bound = -1; t_sample = [];
        return 
    else
        [upper_bound,~] = saturated_interval_stabbing(intervals_upper,ids_upper,sat_buffer,100);
    end
end

