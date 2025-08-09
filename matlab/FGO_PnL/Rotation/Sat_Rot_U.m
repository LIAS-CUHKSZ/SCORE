%%%%
% find the upper bound for Sat-CM rotation problem in PnL 
% with rotation axis confined in a cube under polar coordinates

%%% Inputs:
% line_pair:            L x 1 data structure, comes from pre-processing. 
% Branch:               4 x 1, the given sub-cube.
% epsilon:              scalar, error tolerance.
% sample_resolution:    scalar, control resolution for interval analysis.
% id:                   L x 1, the belonging 2D line id for association.
% sat_buffer:           M x N, storing saturation function value

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% License: MIT

function [upper_bound] = Sat_Rot_U(line_pair,Branch,epsilon,sample_resolution,id,sat_buffer)
N = line_pair.size;
% calculate the extreme values for the h1 and h2 function
% sample_resolution controls the precision 
[h1_upper,h1_lower] = h1_interval_mapping(line_pair,Branch,sample_resolution);
[h2_upper,h2_lower] = h2_interval_mapping(line_pair,Branch,sample_resolution);

% calculate params based on the extreme values
[A_lower,phi_lower,const_lower] = cal_params(line_pair.inner_product,h1_lower,h2_lower);
[A_upper,phi_upper,const_upper] = cal_params(line_pair.inner_product,h1_upper,h2_upper);

% prepare intervals to be satbbed
intervals_upper = []; ids_upper=[];
for i = 1:N
    % calculate interval
    [tmp_interval] = upper_interval(A_upper(i),phi_upper(i),const_upper(i),A_lower(i),phi_lower(i),const_lower(i),epsilon);

    % append the vector
    intervals_upper=[intervals_upper;tmp_interval];

    % if multiple intervals appended, accordingly replicate the 2D line id
    ids_upper = [ids_upper;id(i)*ones(length(tmp_interval)/2,1)];
end
%
if isempty(ids_upper)
    % no interval to be stabbed
    upper_bound = 0;
else
    % saturated interval stabbing to get upper bound
    [upper_bound, ~] = saturated_interval_stabbing(intervals_upper,ids_upper,sat_buffer,100);
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