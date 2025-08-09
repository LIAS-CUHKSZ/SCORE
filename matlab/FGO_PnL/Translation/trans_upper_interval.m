%%% prepare intervals for upper bounds in transltion problem 
% calculate the interval of x which let
% |n_2D_rot'(p_3D-t)|<=epsilon_t
% for any [ty;tz] belongs to branch_yz

%%% inputs
% n_2D_rot  :  1 x 3,  rotated normal vector of the 2D line plane
% p_3D      :  1 x 3,  a point on the 3D line
% epsilon_t :  scalar, tolerance term
% x_limit   :  scalar, x within [0,x_limit]
% vertices  :  2 x 4,  four vertices of branch_yz

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT

function [interval] = trans_upper_interval(n_2D_rot,p_3D,epsilon_t,x_limit,vertices)
    % regularize n_x>=0
    n_x = n_2D_rot(1);
    if n_x<0 
        n_2D_rot = -n_2D_rot;
        n_x = -n_x;
    end

    % solve linear progamming by comparing values at all vertices
    n_yz = n_2D_rot(2:3);
    max_v = -inf; min_v = inf;
    for i=1:4
        vertex = vertices(i,:);
        value = -n_yz*vertex';
        max_v = max(value,max_v);
        min_v = min(value,min_v);
    end
    const = n_2D_rot*p_3D';
    const_max = const+max_v+epsilon_t;
    const_min = const+min_v-epsilon_t;

    % n_x*x-const_max <=0 & nx*x-const_min >=0
    if n_x ==0
        if const_max >=0 && const_min <=0
            interval=[0;x_limit];
        else
            interval=[];
        end
    else
        u_ = min(const_max/n_x, x_limit);
        l_ = max(const_min/n_x, 0);
        if u_<0 || l_>x_limit
            interval=[];
        else
            interval = [l_;u_];
        end
    end

end

