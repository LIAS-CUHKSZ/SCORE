%%%%
% Pre-compute useful quantities for each 2D/3D association
%%% Inputs:
% vector_v: N x 3, the direction vector for each matched 3D map line.
% vector_n: N x 3, the normal vector paramater for each 2D image line.

%%% Author: Xiang Zheng   <224045013@link.cuhk.edu.cn>
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
outer_product_belong = zeros(N,1); 
for i=1:N
    n = vector_n(i,:);
    v = vector_v(i,:);
    outer_product(i,:) = cross(v,n);
    outer_product_belong(i) = outer_product(i,2)>=0; % belonging half sphere, 1 for east and 0 for west
    if outer_product_belong(i)
        vector_outer_east(i,:) =outer_product(i,:);
        vector_outer_west(i,:) =-outer_product(i,:);
    else
        vector_outer_east(i,:) =-outer_product(i,:);
        vector_outer_west(i,:) = outer_product(i,:);
    end
    %
    outer_angle  = zeros(2,1);
    [outer_angle(1), outer_angle(2) ]= xyz_2_polar(outer_product(i,:));
    if outer_angle(2) > pi   % alpha in [0,pi], phi in [0, 2*pi]
        outer_east(i,:) = [pi-outer_angle(1),outer_angle(2)-pi];
        outer_west(i,:) = [outer_angle(1),outer_angle(2)];           
    else
        outer_east(i,:) = [outer_angle(1),outer_angle(2)];
        outer_west(i,:) = [pi-outer_angle(1),outer_angle(2)+pi];
    end
    %
    inner_product(i) = dot(v,n);
    [normal_east(i,:),normal_west(i,:),o_normal_east(i,:),o_normal_west(i,:)] = normal(n,v);
    vector_normal_east(i,:) = polar_2_xyz(normal_east(i,1),normal_east(i,2))';
    vector_normal_west(i,:) = polar_2_xyz(normal_west(i,1),normal_west(i,2))';
    vector_o_normal_east(i,:) = polar_2_xyz(o_normal_east(i,1),o_normal_east(i,2))';
    vector_o_normal_west(i,:) = polar_2_xyz(o_normal_west(i,1),o_normal_west(i,2))';
    outer_norm(i) = norm(outer_product(i,:));
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

function [normal_east,normal_west,o_normal_east,o_normal_west] = normal(v1,v2)
    mid = (v1+v2)/2;
    if(norm(mid)<1e-4)
        normal_east = [0,0];
        normal_west = [0,0];
        [alpha_v1, phi_v1] = xyz_2_polar(v1);
        if(phi_v1>pi)
            o_normal_east = [pi-alpha_v1, phi_v1-pi];
            o_normal_west = [alpha_v1, phi_v1];

        else
            o_normal_east = [alpha_v1, phi_v1];
            o_normal_west = [pi-alpha_v1, phi_v1+pi];
        end

        return;
    end
    mid = mid/norm(mid);
    n =cross(v1,v2);
    n = n/norm(n);
    orthogonal = cross(mid,n);
    orthogonal = orthogonal/norm(orthogonal);
    [alpha_mid, phi_mid] = xyz_2_polar(mid);
    [alpha_orthogonal, phi_orthogonal] = xyz_2_polar(orthogonal);
    if(phi_mid>pi)
        normal_east = [pi-alpha_mid, phi_mid-pi];
        normal_west = [alpha_mid, phi_mid];
    else
        normal_east = [alpha_mid, phi_mid];
        normal_west = [pi-alpha_mid, phi_mid+pi];
    end
    if(phi_orthogonal>pi)
        o_normal_east = [pi-alpha_orthogonal, phi_orthogonal-pi];
        o_normal_west = [alpha_orthogonal, phi_orthogonal];
    else
        o_normal_east = [alpha_orthogonal, phi_orthogonal];
        o_normal_west = [pi-alpha_orthogonal, phi_orthogonal+pi];
    end
end

