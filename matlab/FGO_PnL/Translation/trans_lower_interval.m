%%% trans_lower_interval
% calculate the interval of x which let
% |n_2D_rot'(p_3D-t)|<=epsilon_t
% for [ty;tz]=yz_sampled

%%% inputs
% n_2D_rot  :  1 x 3
% p_3D      :  1 x 3
% epsilon_t :  scalar
% yz_sampled:  2 x 1
% x_limit   :  scalar

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
 
%%% License: MIT
%%%%
function interval = trans_lower_interval(n_2D_rot,p_3D,epsilon_t,yz_sampled,x_limit)
    interval = [];
    nx = n_2D_rot(1);
    if nx<0
       n_2D_rot = -n_2D_rot; % regularize and let nz>=0
       nx       = -nx;
    end
    const = n_2D_rot*p_3D'-n_2D_rot(2:3)*yz_sampled;
    %%%
    if nx==0
        if abs(const)<=epsilon_t
            interval = [0;x_limit];
        else
            interval = [];
        end
    else
        interval = [const-epsilon_t;const+epsilon_t]/nx;
        if interval(2)<0 || interval(1)>x_limit
            interval=[];
        else
            interval(1) = max(0,interval(1));
            interval(2) = min(x_limit,interval(2));
        end
    end
end

