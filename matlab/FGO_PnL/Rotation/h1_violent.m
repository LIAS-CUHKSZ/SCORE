%%%%
% Calculate extreme values for h1 within a given sub-cube by tranversing
% the boundary, used for debugging h1_interval_mapping.m
% h1(u,v,n)  = u'(v \times n)

%%% Inputs:
% line_pair:            data structure, comes from pre-processing.
% Branch:               4 x 1, the given sub-cube.
% sample_resolution:    scalar, control resolution for interval analysis.

%%% Author:  Haodong JIANG <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [upper,lower] =h1_violent(line_pair,branch,sample_resolution)
N = line_pair.size;
upper = zeros(N,1);
lower = zeros(N,1);
if branch(3)-branch(1)>=sample_resolution
    alpha_grid=branch(1):sample_resolution:branch(3);
    phi_grid = branch(2):sample_resolution:branch(4);
else
    alpha_grid = [branch(1),branch(3)];
    phi_grid   = [branch(2),branch(4)];
end
M = length(alpha_grid);
boundary = zeros(2,M*4-4);
boundary(:,1:M)=[alpha_grid(1)*ones(1,M);phi_grid];
boundary(:,M:2*M-1)=[alpha_grid;phi_grid(end)*ones(1,M)];
boundary(:,2*M-1:3*M-2)=[alpha_grid(end)*ones(1,M);phi_grid(end:-1:1)];
boundary(:,3*M-2:4*M-4)=[alpha_grid(end:-1:2);phi_grid(1)*ones(1,M-1)];
MM = size(boundary,2);
for n =1:N
    u_ = nan; l_ =nan;
    if line_pair.outer_product_belong(n) % belongs to the east sphere
        alpha_l = line_pair.outer_west(n,1); phi_l = line_pair.outer_west(n,2);
        alpha_u = line_pair.outer_east(n,1); phi_u = line_pair.outer_east(n,2);
    else
        alpha_l = line_pair.outer_east(n,1); phi_l = line_pair.outer_east(n,2);
        alpha_u = line_pair.outer_west(n,1); phi_u = line_pair.outer_west(n,2);
    end
    if alpha_l>=branch(1) && alpha_l<=branch(3) && phi_l>=branch(2) && phi_l<=branch(4)
        l_ = -line_pair.outer_norm(n);
    end
    if alpha_u>=branch(1) && alpha_u<=branch(3) && phi_u>=branch(2) && phi_u<=branch(4)
        u_ = line_pair.outer_norm(n);
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


