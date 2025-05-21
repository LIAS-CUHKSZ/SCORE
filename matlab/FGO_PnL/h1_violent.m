%%%%
% Calculate extreme values for h1 within a given sub-cube.
% h1(u,v,n)  = u'(v \times n)

%%% Inputs:
% line_pair:            data structure, comes from pre-processing.
% Branch:               4 x 1, the given sub-cube.
% sample_resolution:    scalar, control resolution for interval analysis.

%%% Author:  Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [upper,lower] =h1_violent(line_pair,branch,sample_resolution)
N = line_pair.size;
upper = zeros(N,1);
lower = zeros(N,1);
alpha_grid=branch(1):sample_resolution:branch(3);
phi_grid = branch(2):sample_resolution:branch(4);
M = length(alpha_grid);
boundary = zeros(2,M*4-4);
boundary(:,1:M)=[alpha_grid(1)*ones(1,M);phi_grid];
boundary(:,M:2*M-1)=[alpha_grid;phi_grid(1)*ones(1,M)];
boundary(:,2*M-1:3*M-2)=[alpha_grid(end)*ones(1,M);phi_grid];
boundary(:,3*M-2:4*M-4)=[alpha_grid(end:-1:2);phi_grid(end)*ones(1,M-1)];
MM = size(boundary,2);
for n =1:N
    cn = line_pair.outer_product(n,:);
    [alpha_l,phi_l]=xyz_2_polar(-cn);
    u_ = nan; l_ =nan;
    if alpha_l>=branch(1) && alpha_l<=branch(3) && phi_l>=branch(2) && phi_l<=branch(4)
        l_ = -1;
    end
    [alpha_u,phi_u]=xyz_2_polar(cn);
    if alpha_u>=branch(1) && alpha_u<=branch(3) && phi_u>=branch(2) && phi_u<=branch(4)
        u_ = 1;
    end
    for m=1:MM
        axis = polar_2_xyz(boundary(1,m),boundary(2,m));
        value = line_pair.outer_product(n,:)*axis;
        u_ = max(u_,value);
        l_ = min(l_,value);
    end
    upper(n)=u_;
    lower(n)=l_;
end
end

function [far,near] = interval_projection(a, interval)
%   a is a given scalar, x falls in a given interval
%  return  far= argmax_x |x-a| , near= argmin_x |x-a| 
    if a <interval(1)
        far = interval(2);
        near = interval(1);
    elseif a<= (interval(1)+interval(2))/2
        far = interval(2);
        near= a;
    elseif a<=interval(2)
        far = interval(1);
        near = a;
    else
        far = interval(1);
        near= interval(2);
    end
end


