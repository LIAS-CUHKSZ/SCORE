%%% trans_lower_interval
% calculate the interval of x which let
% |n_2D_rot'(p_3D-t)|<=epsilon_t
% for [ty;tz] belongs to the branch

%%% inputs
% n_2D_rot  :  1 x 3
% p_3D      :  1 x 3
% epsilon_t :  scalar
% x_limit   :  scalar
% branch_yz :  4 x 1
% vertices  :  2 x 4

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT

function [interval] = trans_upper_interval(n_2D_rot,p_3D,epsilon_t,x_limit,vertices)
    n_x = n_2D_rot(1);
    if n_x<0 %regularization
        n_2D_rot = -n_2D_rot;
        n_x = -n_x;
    end
    n_yz = n_2D_rot(2:3);
    max_v = -10000; min_v = 10000;
    for i=1:4
        vertex = vertices(i,:);
        value = -n_yz*vertex';
        max_v = max(value,max_v);
        min_v = min(value,min_v);
    end
    const = n_2D_rot*p_3D';
    const_max = const+max_v+epsilon_t;
    const_min = const+min_v-epsilon_t;
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

