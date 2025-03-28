function [upper , lower] = h2_interval_mapping(line_pair,branch,sample_resolution)
    N = line_pair.size;
    upper = zeros(N,1);
    lower = zeros(N,1);
    cube_width = branch(3)-branch(1);
    maximum = 0;
    minimum = 0;
    if cube_width <=sample_resolution
        vertex_cache=zeros(4,3);
        vertex_cache(1,:)= polar_2_xyz(branch(1),branch(2));
        vertex_cache(2,:)= polar_2_xyz(branch(1),branch(4));
        vertex_cache(3,:)= polar_2_xyz(branch(3),branch(2));
        vertex_cache(4,:)= polar_2_xyz(branch(3),branch(4));
    else 
        alpha = branch(1):sample_resolution:branch(3);
        phi = branch(2):sample_resolution:branch(4);
        temp = length(alpha)-1;
        % temp
        vertex_cache=zeros(temp*4,3);
        vertex_cache(1:temp,:)=vec_polar2xyz(alpha(1:temp),phi(1));
        vertex_cache(temp+1:2*temp,:) = vec_polar2xyz(alpha(end),phi(1:temp));
        vertex_cache(2*temp+1:3*temp,:) = vec_polar2xyz(alpha(2:end),phi(end));
        vertex_cache(3*temp+1:4*temp,:)=vec_polar2xyz(alpha(1),phi(2:end));

    end
    for i = 1:N
        n_i = line_pair.vector_n(i,:);
        v_i = line_pair.vector_v(i,:);
        inner_product= line_pair.inner_product(i);
        if branch(2)<pi
            normal_angle = line_pair.normal_east(i,:);
            normal_vector = line_pair.vector_normal_east(i,:);
            o_normal_angle = line_pair.o_normal_east(i,:);
            o_normal_vector = line_pair.vector_o_normal_east(i,:);
        else
            normal_angle = line_pair.normal_west(i,:);
            normal_vector = line_pair.vector_normal_west(i,:);
            o_normal_angle = line_pair.o_normal_west(i,:);
            o_normal_vector = line_pair.vector_o_normal_west(i,:);
        end
        flag = (normal_angle(1)>=branch(1) && normal_angle(1)<=branch(3) && normal_angle(2)>=branch(2) && normal_angle(2)<=branch(4)) *2 + (o_normal_angle(1)>=branch(1) && o_normal_angle(1)<=branch(3) && o_normal_angle(2)>=branch(2) && o_normal_angle(2)<=branch(4));
        switch flag
            case 3
                maximum = dot(normal_vector,n_i)*dot(normal_vector,v_i) ;
                minimum = dot(o_normal_vector,n_i)*dot(o_normal_vector,v_i) ;
                upper(i) = maximum-inner_product;
                lower(i) = minimum-inner_product;
                continue;
            case 2
                maximum = dot(normal_vector,n_i)*dot(normal_vector,v_i) ;

            case 1 
                minimum = dot(o_normal_vector,n_i)*dot(o_normal_vector,v_i) ;
        end
        
        ttt = (vertex_cache*n_i').*(vertex_cache*v_i');
        [tmp_minimum,tmp_maximum] = bounds(ttt,"all");
        if flag==2
            upper(i) = maximum;
            lower(i) = tmp_minimum;
        elseif flag==1
            upper(i) = tmp_maximum;
            lower(i) = minimum;
        else
            upper(i) = tmp_maximum;
            lower(i) = tmp_minimum;
        end
        upper(i) = upper(i)- inner_product;
        lower(i) = lower(i) - inner_product;

    end


end



function output = vec_polar2xyz(alpha,phi)
    % output  N*3
    n = max(length(alpha),length(phi));
    
    output = zeros(n,3);
    as = sin(alpha');
    output(:,3) = cos(alpha');
    output(:,2) = as.*sin(phi');
    output(:,1) = as.*cos(phi');
end