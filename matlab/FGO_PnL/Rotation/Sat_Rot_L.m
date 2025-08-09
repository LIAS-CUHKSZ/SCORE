%%%%
% find the lower bound for Sat-CM rotation problem in PnL 
% with rotation axis confined in a cube under polar coordinates

%%% Inputs:
% line_pair:            L x 1 data structure, comes from pre-processing. 
% Branch:               4 x 1, the given sub-cube.
% epsilon:              scalar, error tolerance.
% id:                   L x 1, the belonging 2D line id for association.
% sat_buffer:           M x N, storing saturation function value
% prox_thres:           scalar, used for clustering proximate stabbers

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% License: MIT

function [lower_bound,theta_opt] = Sat_Rot_L(line_pair,Branch,epsilon,id,sat_buffer,prox_thres)
N = line_pair.size;
% take the center point
u_center = polar_2_xyz(0.5*(Branch(1)+Branch(3)),0.5*(Branch(2)+Branch(4)));

% calculate for each association the values of functions h1 h2 at the center point 
h1_center = zeros(N,1); h2_center = zeros(N,1);
for i = 1:N
    h1_center(i) = dot(u_center,line_pair.outer_product(i,:));
    h2_center(i) = dot(u_center,line_pair.vector_n(i,:))*dot(u_center,line_pair.vector_v(i,:))-line_pair.inner_product(i);
end

% calculate params based on the function values
[A_center,phi_center,const_center] = cal_params(line_pair.inner_product,h1_center,h2_center);

% prepare intervals to be stabbed
intervals_lower = []; ids_lower=[];
for i = 1:N
    % calculate interval
    [tmp_interval] = lower_interval(A_center(i),phi_center(i),const_center(i),epsilon);

    % append the vector storing intervals
    intervals_lower=[intervals_lower;tmp_interval];
    
    % if multiple intervals appended, accordingly replicate the 2D line id
    ids_lower = [ids_lower;id(i)*ones(length(tmp_interval)/2,1)];
end
%
if isempty(ids_lower)
    % no intervals to be stabbed
    lower_bound = 0;
    theta_opt = 0;
else
    % saturated interval stabbing to get lower bound
    [lower_bound, theta_opt] = saturated_interval_stabbing(intervals_lower,ids_lower,sat_buffer,prox_thres);
end
end

function [A,phi,const] = cal_params(product, h1 ,h2)
    % f =  product + sin(theta)* h1 + (1-cos theta )* h2  
    %   =  h1 sin(theta) - h2 cos(theta) + product + h2
    %   =  A·sin(theta+phi)+const
    A = sqrt(h1.^2 + h2.^2);
    phi = atan2(-h2,h1);
    phi =phi.*(phi>=0) + (phi+2*pi).*(phi<0); % make sure that phi in [0,2*pi]
    const = product+h2;
end