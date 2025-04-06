%%%%
% Implementation of the FGO rotation estimator under saturated consensus maximization
%%% Inputs:
% vector_v: N x 3, the direction vector for each matched 3D map line.
% vector_n: N x 3, the normal vector paramater for each 2D image line.
% branch_resolution: scalar, stop bnb when cube length < resolution.
% epsilon: scalar, for Sat-CM formulation.
% sample_resolution: scalar, control resolution for interval analysis.
% verbose_flag: bool, set true for detailed bnb process info.
% id: N x 1, the belonging 2D line id for each matched pair.
% kernel: function, the saturation function.
%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%
function [Q_upper,Q_lower,theta_lower]=Sat_Bounds_FGO(line_pair,Branch,epsilon,sample_resolution,id,kernel)
[h1_upper,h1_lower] = h1_interval_mapping(line_pair,Branch,sample_resolution);
[h2_upper,h2_lower] = h2_interval_mapping(line_pair,Branch,sample_resolution);
N = line_pair.size;
[A_lower,phi_lower,const_lower] = generate_params(line_pair.inner_product,h1_lower,h2_lower);
[A_upper,phi_upper,const_upper] = generate_params(line_pair.inner_product,h1_upper,h2_upper);
h1_center = zeros(N,1);
h2_center = zeros(N,1);
u_center = polar_2_xyz(0.5*(Branch(1)+Branch(3)),0.5*(Branch(2)+Branch(4)));
for i = 1:N
    h1_center(i) = dot(u_center,line_pair.outer_product(i,:));
    h2_center(i) = dot(u_center,line_pair.vector_n(i,:))*dot(u_center,line_pair.vector_v(i,:))-line_pair.inner_product(i);
end
[A_center,phi_center,const_center] = generate_params(line_pair.inner_product,h1_center,h2_center);
intervals_upper = []; ids_upper=[];
intervals_lower = []; ids_lower=[];
for i = 1:N
    %%%
    [tmp_interval] = upper_interval(A_upper(i),phi_upper(i),const_upper(i),A_lower(i),phi_lower(i),const_lower(i),epsilon);
    intervals_upper=[intervals_upper;tmp_interval];
    ids_upper = [ids_upper;id(i)*ones(length(tmp_interval)/2,1)];
    %%%
    [tmp_interval] = lower_interval(A_center(i),phi_center(i),const_center(i),epsilon);
    intervals_lower=[intervals_lower;tmp_interval];
    ids_lower = [ids_lower;id(i)*ones(length(tmp_interval)/2,1)];
end
[Q_lower, theta_lower] = saturated_interval_stabbing(intervals_lower,ids_lower,kernel);
[Q_upper, ~] = saturated_interval_stabbing(intervals_upper,ids_upper,kernel);
end
%%%%%%%%%%%%%%%%%%%%%%%%
function [A,phi,const] = generate_params(product, h1 ,h2)
    %%% f =  c + sin(theta)* h1 + (1-cos theta )* h2  
    %%%   =  h1 sin(theta) - h2 cos(theta) + c+ h2
    A = sqrt(h1.^2 + h2.^2);
    phi = atan2(-h2,h1);
    phi =phi.*(phi>=0) + (phi+2*pi).*(phi<0);
    const = product+h2;
end