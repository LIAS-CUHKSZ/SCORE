%%%%
% Calculate lower and upper bound for a given sub-cube.

%%% Inputs:
% line_pair:            L x 1 data structure, comes from pre-processing. 
% Branch:               4 x 1, the given sub-cube.
% epsilon:              scalar, error tolerance.
% sample_resolution:    scalar, control resolution for interval analysis.
% id:                   L x 1, the belonging 2D line id for association.
% kernel_buffer:        M x N, storing saturation function value
% prox_thres:           scalar, used to cluster similar stabbers

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% License: MIT

%%%%
function [Q_upper,Q_lower,theta_lower]=Sat_Bounds_FGO(line_pair,Branch,epsilon,sample_resolution,id,kernel_buffer,prox_thres)
N = line_pair.size;
% -----------------------------------------------------------
% --- 1. calculate lower bound by taking the center point ---
u_center = polar_2_xyz(0.5*(Branch(1)+Branch(3)),0.5*(Branch(2)+Branch(4)));
% calculate for each association:
% the values of functions h1 h2 at the center point 
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
    Q_lower = 0;
    theta_lower = 0;
else
    % saturated interval stabbing to get lower bound
    [Q_lower, theta_lower] = saturated_interval_stabbing(intervals_lower,ids_lower,kernel_buffer,prox_thres);
end

% -----------------------------------------------------------
% --- 2. calculate upper bound based on rigorous interval anlaysis

% calculate the extreme values for the h1 and h2 function
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

    % append the vector storing intervals
    intervals_upper=[intervals_upper;tmp_interval];

    % if multiple intervals appended, accordingly replicate the 2D line id
    ids_upper = [ids_upper;id(i)*ones(length(tmp_interval)/2,1)];
end
%
if isempty(ids_upper)
    % no interval to be stabbed
    Q_upper = 0;
else
    % saturated interval stabbing to get upper bound
    [Q_upper, ~] = saturated_interval_stabbing(intervals_upper,ids_upper,kernel_buffer,prox_thres);
end
end

%% sub-functions
%%%
function [A,phi,const] = cal_params(product, h1 ,h2)
    % f =  product + sin(theta)* h1 + (1-cos theta )* h2  
    %   =  h1 sin(theta) - h2 cos(theta) + product + h2
    %   =  A·sin(theta+phi)+const
    A = sqrt(h1.^2 + h2.^2);
    phi = atan2(-h2,h1);
    phi =phi.*(phi>=0) + (phi+2*pi).*(phi<0); % make sure that phi in [0,2*pi]
    const = product+h2;
end

%%% for debugging
% [h1_upper,h1_lower]=h1_violent(line_pair,Branch,sample_resolution);
% for i = 1:length(h1_upper)
%     if abs(h1_upper_(i)-h1_upper(i))>0.01
%         fprintf('UPPER: branch: [%f , %f, %f , %f ], line_id: %d, h1_violent: %f, h1_func: %f\n ',Branch(1),Branch(2),Branch(3),Branch(4),i , h1_upper(i),h1_upper_(i)); % 保留2位小数
%     end
%     if abs(h1_lower_(i)-h1_lower(i))>0.01
%         fprintf('LOWER: branch: [%f , %f, %f , %f ], line_id: %d, h1_violent: %f, h1_func: %f\n ',Branch(1),Branch(2),Branch(3),Branch(4),i , h1_lower(i),h1_lower_(i)); % 保留2位小数
%     end
% end