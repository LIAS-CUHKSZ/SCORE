%%%%
% Pre-compute useful quantities for each pair of matched 2D/3D lines.

%%% Inputs:
% vector_v: N x 3, the direction vector for each matched 3D map line.
% vector_n: N x 3, the normal vector paramater for each 2D image line.

%%% Author: Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [line_pair_data] = data_process(vector_n,vector_v)
N= size(vector_n,1);
outer_product=zeros(N,3);
outer_east = zeros(N,2);
outer_west = zeros(N,2);
inner_product=zeros(N,1);
normal_east = zeros(N,2);
normal_west = zeros(N,2);
o_normal_east = zeros(N,2);
o_normal_west = zeros(N,2);
vector_normal_east = zeros(N,3);
vector_normal_west = zeros(N,3);
vector_o_normal_east = zeros(N,3);
vector_o_normal_west = zeros(N,3);
vector_outer_west =zeros(N,3);
vector_outer_east= zeros(N,3);
outer_norm =zeros(N,1);
outer_product_belong = zeros(N,1); % which half sphere the vector belongs to  1 for east and 0 for west
for i=1:N
    n = vector_n(i,:);
    v = vector_v(i,:);
    outer_product(i,:) = cross(v,n);
    outer_angle  = zeros(2,1);
    [outer_angle(1), outer_angle(2) ]= xyz_2_polar(outer_product(i,:));
    if outer_angle(2) > pi
        outer_east(i,:) = [pi-outer_angle(1),outer_angle(2)-pi];
        outer_west(i,:) = [outer_angle(1),outer_angle(2)];
    else
        outer_east(i,:) = [outer_angle(1),outer_angle(2)];
        outer_west(i,:) = [pi-outer_angle(1),outer_angle(2)+pi];
    end
    inner_product(i) = dot(n,v);
    [normal_east(i,:),normal_west(i,:),o_normal_east(i,:),o_normal_west(i,:)] = normal(n,v);
    vector_normal_east(i,:) = [sin(normal_east(i,1))*cos(normal_east(i,2)),sin(normal_east(i,1))*sin(normal_east(i,2)),cos(normal_east(i,1))];
    vector_normal_west(i,:) = [sin(normal_west(i,1))*cos(normal_west(i,2)),sin(normal_west(i,1))*sin(normal_west(i,2)),cos(normal_west(i,1))];
    vector_o_normal_east(i,:) = [sin(o_normal_east(i,1))*cos(o_normal_east(i,2)),sin(o_normal_east(i,1))*sin(o_normal_east(i,2)),cos(o_normal_east(i,1))];
    vector_o_normal_west(i,:) = [sin(o_normal_west(i,1))*cos(o_normal_west(i,2)),sin(o_normal_west(i,1))*sin(o_normal_west(i,2)),cos(o_normal_west(i,1))];
    outer_norm(i) = norm(outer_product(i,:));
    if outer_product(i,2) >= 0
        outer_product_belong(i) = 1;
        vector_outer_east(i,:) =outer_product(i,:);
        vector_outer_west(i,:) =-outer_product(i,:);
    else
        vector_outer_east(i,:) =-outer_product(i,:);
        vector_outer_west(i,:) = outer_product(i,:);
        outer_product_belong(i) = 0;
    end
end
line_pair_data.outer_product_belong = outer_product_belong;
line_pair_data.vector_normal_east = vector_normal_east;
line_pair_data.vector_normal_west = vector_normal_west;
line_pair_data.vector_o_normal_east = vector_o_normal_east;
line_pair_data.vector_o_normal_west = vector_o_normal_west;
line_pair_data.inner_product = inner_product;
line_pair_data.outer_product = outer_product;
line_pair_data.vector_outer_east = vector_outer_east;
line_pair_data.vector_outer_west = vector_outer_west;
line_pair_data.normal_east = normal_east;
line_pair_data.normal_west = normal_west;
line_pair_data.o_normal_east = o_normal_east;
line_pair_data.o_normal_west = o_normal_west;
line_pair_data.vector_n = vector_n;
line_pair_data.vector_v = vector_v;
line_pair_data.size = N;
line_pair_data.outer_east = outer_east;
line_pair_data.outer_west = outer_west;
line_pair_data.outer_norm =outer_norm;
end

