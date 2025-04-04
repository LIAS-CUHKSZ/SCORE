function [outer_product,outer_east,outer_west,  inner_product, normal_east, normal_west,  o_normal_east,  o_normal_west] = data_process(vector_n,vector_v)
    %% 应该对角平分线进行标记，对半球归属进行向量的标记
    N= size(vector_n,1);
    outer_product=zeros(N,3);
    outer_east = zeros(N,2);
    outer_west = zeros(N,2);
    inner_product=zeros(N,1);
    normal_east = zeros(N,2);
    normal_west = zeros(N,2);
    o_normal_east = zeros(N,2);
    o_normal_west = zeros(N,2);
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
    end
    
end

