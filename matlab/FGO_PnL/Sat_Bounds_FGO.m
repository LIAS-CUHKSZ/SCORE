%%%%
% Calculate lower and upper bound for a given sub-cube.

%%% Inputs:
% line_pair:            data structure, comes from pre-processing. 
% Branch:               4 x 1, the given sub-cube.
% epsilon:              scalar, used in Sat-CM formulation.
% sample_resolution:    scalar, control resolution for interval analysis.
% id:                   N x 1, the belonging 2D line id for each matched pair.
% kernel:               function, saturation function.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT

%%%%
function [Q_upper,Q_lower,theta_lower]=Sat_Bounds_FGO(line_pair,Branch,epsilon,sample_resolution,id,kernel_buffer)
N = line_pair.size;
%%% calculate lower bound by taking the center point
% obtain the interval for each matched pair
u_center = polar_2_xyz(0.5*(Branch(1)+Branch(3)),0.5*(Branch(2)+Branch(4)));
h1_center = zeros(N,1);
h2_center = zeros(N,1);
for i = 1:N
    h1_center(i) = dot(u_center,line_pair.outer_product(i,:));
    h2_center(i) = dot(u_center,line_pair.vector_n(i,:))*dot(u_center,line_pair.vector_v(i,:))-line_pair.inner_product(i);
end
[A_center,phi_center,const_center] = cal_params(line_pair.inner_product,h1_center,h2_center);
intervals_lower = []; ids_lower=[];
for i = 1:N
    [tmp_interval] = lower_interval(A_center(i),phi_center(i),const_center(i),epsilon);
    intervals_lower=[intervals_lower;tmp_interval];
    ids_lower = [ids_lower;id(i)*ones(length(tmp_interval)/2,1)];
end
if isempty(ids_lower)
    Q_lower = 0;
    theta_lower = 0;
else
    [Q_lower, theta_lower] = saturated_interval_stabbing(intervals_lower,ids_lower,kernel_buffer);
end
%%% calculate upper bound based on rigoros interval anlaysis
% obtain the interval for each matched pair
% calculate the extreme values for the h1 and h2 function
[h1_upper_,h1_lower_] = h1_interval_mapping(line_pair,Branch,sample_resolution);
[h1_upper,h1_lower]=h1_violent(line_pair,Branch,sample_resolution);
if sum(abs(h1_upper_-h1_upper))>0.01
    h1_upper_'
    h1_upper'
end
if sum(abs(h1_lower_-h1_lower))>0.01
    h1_lower_'
    h1_lower'
end
[h2_upper,h2_lower] = h2_interval_mapping(line_pair,Branch,sample_resolution);
[A_lower,phi_lower,const_lower] = cal_params(line_pair.inner_product,h1_lower,h2_lower);
[A_upper,phi_upper,const_upper] = cal_params(line_pair.inner_product,h1_upper,h2_upper);
intervals_upper = []; ids_upper=[];
for i = 1:N
    [tmp_interval] = upper_interval(A_upper(i),phi_upper(i),const_upper(i),A_lower(i),phi_lower(i),const_lower(i),epsilon);
    intervals_upper=[intervals_upper;tmp_interval];
    ids_upper = [ids_upper;id(i)*ones(length(tmp_interval)/2,1)];
end
%
if isempty(ids_upper)
    Q_upper = 0;
else
    [Q_upper, ~] = saturated_interval_stabbing(intervals_upper,ids_upper,kernel_buffer);
end
end

%%%
function [A,phi,const] = cal_params(product, h1 ,h2)
    %%% f =  product + sin(theta)* h1 + (1-cos theta )* h2  
    %%%   =  h1 sin(theta) - h2 cos(theta) + product + h2
    A = sqrt(h1.^2 + h2.^2);
    phi = atan2(-h2,h1);
    phi =phi.*(phi>=0) + (phi+2*pi).*(phi<0); % make sure that phi in [0,2*pi]
    const = product+h2;
end

