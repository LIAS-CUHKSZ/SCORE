%%%%
% Calculate extreme values for h1 within a given sub-cube.
% h1(u,v,n)  = u'(v \times n)

%%% Inputs:
% line_pair:            data structure, comes from pre-processing.
% Branch:               4 x 1, the given sub-cube.
% sample_resolution:    scalar, control resolution for interval analysis.

%%% Author:  Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [upper,lower] =h1_interval_mapping(line_pair,branch,sample_resolution)
N = line_pair.size;
upper = zeros(N,1);
lower = zeros(N,1);
cube_width = branch(3)-branch(1);
range_alpha= [branch(1),branch(3)];
range_phi = [branch(2),branch(4)];
if cube_width<=sample_resolution
    for i = 1:N
        east = line_pair.outer_product_belong(i);
        if (range_phi(2) > pi && east == 0) || (range_phi(2) <= pi && east == 1)
            flag =1;
        else
            flag =-1;
        end
        if range_phi(1) <= pi&& range_phi(2)<=pi
            outer_alpha = line_pair.outer_east(i,1);
            outer_phi = line_pair.outer_east(i,2);
            x =line_pair.vector_outer_east(i,:);
        else
            outer_alpha = line_pair.outer_west(i,1);
            outer_phi = line_pair.outer_west(i,2);
            x =line_pair.vector_outer_west(i,:);
        end
        [phi_far,phi_near]     = interval_projection(outer_phi,range_phi);
        [alpha_far,alpha_near] = interval_projection(outer_alpha,range_alpha);
        %%% find_maximum
        delta_phi_near = abs(phi_near - outer_phi);
        if delta_phi_near ==0
            maximum = dot(x,polar_2_xyz(alpha_near,phi_near));
        else
            maximum = max(  dot(x,polar_2_xyz(range_alpha(1),phi_near)),...
                dot(x,polar_2_xyz(range_alpha(2),phi_near)));
        end
        %%% find_minimum
        minimum = min(  dot(x,polar_2_xyz(range_alpha(1),phi_far)),...
            dot(x,polar_2_xyz(range_alpha(2),phi_far)));
        if flag == 1
            upper(i) = maximum;
            lower(i) = minimum;
        else
            upper(i)= -minimum;
            lower(i) = -maximum;
        end
    end
else % cube_width > sample_resolution
    for i = 1:N
        east = line_pair.outer_product_belong(i);
        if (range_phi(2) > pi && east == 0) || (range_phi(2) <= pi && east == 1)
            flag =1;
        else
            flag =-1;
        end
        if range_phi(1) <= pi&& range_phi(2)<=pi
            outer_alpha = line_pair.outer_east(i,1);
            outer_phi = line_pair.outer_east(i,2);
            x =line_pair.vector_outer_east(i,:);
        else
            outer_alpha = line_pair.outer_west(i,1);
            outer_phi = line_pair.outer_west(i,2);
            x =line_pair.vector_outer_west(i,:);
        end
        [phi_far,phi_near]     = interval_projection(outer_phi,range_phi);
        [alpha_far,alpha_near] = interval_projection(outer_alpha,range_alpha);
        is_north = range_alpha(1) <= pi/2 && range_alpha(2)<=pi/2;
        is_south = ~is_north;
        %%% find_maximum
        delta_phi_near = abs(phi_near - outer_phi);
        if abs(outer_alpha-pi/2)<1e-5 && (range_alpha(1)>=pi/2 || range_alpha(2)<=pi/2)
            if (delta_phi_near<=pi/2&& is_north) ||(delta_phi_near>pi/2&& is_south)
                maximum = dot(x,polar_2_xyz(range_alpha(2),phi_near));
            else
                maximum = dot(x,polar_2_xyz(range_alpha(1),phi_near));
            end
        elseif delta_phi_near ==0
            maximum = dot(x,polar_2_xyz(alpha_near,phi_near));
        elseif delta_phi_near>pi/2
            tangent = tan(outer_alpha)*cos(delta_phi_near);
            if tangent>1e8
                max_alpha = pi/2;
            else
                max_alpha = atan(tangent);
                if(max_alpha<0)
                    max_alpha = max_alpha+pi;
                end
            end
            if max_alpha<=sum(range_alpha)/2
                maximum = dot(x,polar_2_xyz(range_alpha(2),phi_near));
            else
                maximum = dot(x,polar_2_xyz(range_alpha(1),phi_near));
            end
        elseif delta_phi_near<pi/2 && outer_alpha<pi/2 && range_alpha(1)>=outer_alpha
            maximum = dot(x,polar_2_xyz(range_alpha(1),phi_near));
        elseif delta_phi_near<pi/2 && outer_alpha>pi/2 && range_alpha(2)<=pi-outer_alpha
            maximum = dot(x,polar_2_xyz(range_alpha(2),phi_near));
        elseif delta_phi_near==pi/2
            if outer_alpha<=pi/2
                maximum = dot(x,polar_2_xyz(range_alpha(1),phi_near));
            else
                maximum = dot(x,polar_2_xyz(range_alpha(2),phi_near));
            end
        else
            tangent = tan(outer_alpha)*cos(delta_phi_near);
            if tangent>1e8
                max_alpha = pi/2;
            else
                max_alpha = atan(tangent);
                if(max_alpha<0)
                    max_alpha = max_alpha+pi;
                end
            end
            if max_alpha<=range_alpha(1)
                maximum = dot(x,polar_2_xyz(range_alpha(1),phi_near));
            elseif max_alpha<=range_alpha(2)
                maximum = dot(x,polar_2_xyz(max_alpha,phi_near));
            else
                maximum = dot(x,polar_2_xyz(range_alpha(2),phi_near));
            end
        end
        %%% find_minimum
        delta_phi_far = abs(phi_far - outer_phi);
        if abs(outer_alpha-pi/2)<1e-5 && (range_alpha(1)>=pi/2 || range_alpha(2)<=pi/2)
            if  (delta_phi_far<=pi/2 && is_north) ||(delta_phi_far>pi/2&& is_south)
                minimum = dot(x,polar_2_xyz(range_alpha(1),phi_far));
            else
                minimum = dot(x,polar_2_xyz(range_alpha(2),phi_far));
            end
        elseif delta_phi_far <pi/2
            tangent = tan(outer_alpha)*cos(delta_phi_far);
            if tangent>1e8
                min_alpha = pi/2;
            else
                min_alpha = atan(tangent);
                if(min_alpha<0)
                    min_alpha = min_alpha+pi;
                end
            end
            if min_alpha<=sum(range_alpha)/2
                minimum = dot(x,polar_2_xyz(range_alpha(2),phi_far));
            else
                minimum = dot(x,polar_2_xyz(range_alpha(1),phi_far));
            end

        elseif delta_phi_far >pi/2 && outer_alpha <pi/2 && range_alpha(2)<=pi-outer_alpha
            minimum = dot(x,polar_2_xyz(range_alpha(2),phi_far));
        elseif delta_phi_far >pi/2 && outer_alpha >pi/2 && range_alpha(1)>=pi-outer_alpha
            minimum = dot(x,polar_2_xyz(range_alpha(1),phi_far));
        elseif delta_phi_far == pi/2
            if outer_alpha<=pi/2
                minimum = dot(x,polar_2_xyz(range_alpha(2),phi_far));
            else
                minimum = dot(x,polar_2_xyz(range_alpha(1),phi_far));
            end
        else
            tangent = tan(outer_alpha)*cos(delta_phi_far);
            if tangent>1e8
                min_alpha = pi/2;
            else
                min_alpha = atan(tangent);
                if(min_alpha<0)
                    min_alpha = min_alpha+pi;
                end
            end
            if min_alpha<=range_alpha(1)
                minimum = dot(x,polar_2_xyz(range_alpha(1),phi_far));
            elseif min_alpha<=range_alpha(2)
                minimum = dot(x,polar_2_xyz(min_alpha,phi_far));
            else
                minimum = dot(x,polar_2_xyz(range_alpha(2),phi_far));
            end
        end
        if flag == 1
            upper(i) = maximum;
            lower(i) = minimum;
        else
            upper(i)= -minimum;
            lower(i) = -maximum;
        end
    end
end


end

function [far,near] = interval_projection(a, interval)
%   a is a given scalar, x falls in a given interval
%  return  far= argmax_x |x-a| , near= argmin_x |x-a| 
    if a <interval(1)
        far = interval(2);
        near = interval(1);
    elseif a<= (interval(1)+interval(2))/2
        far = interval(2);
        near= a;
    elseif a<=interval(2)
        far = interval(1);
        near = a;
    else
        far = interval(1);
        near= interval(2);
    end
end


