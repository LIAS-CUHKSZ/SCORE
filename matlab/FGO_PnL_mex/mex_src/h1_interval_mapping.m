function [upper,lower] =h1_interval_mapping(line_pair,branch,sample_resolution)
%H1_BOUNDS 此处显示有关此函数的摘要
%%% h1(u,v,n)  = u'(v \times n)
%%% upper: N x 1
%%% lower: N x 1
    N = line_pair.size;
    upper = zeros(N,1);
    lower = zeros(N,1);
    cube_width = branch(3)-branch(1);
    range_alpha= [branch(1),branch(3)];
    range_phi = [branch(2),branch(4)];
    if cube_width<=sample_resolution
        for i = 1:N
            east = line_pair.outer_product_belong(i);
            %  flag =1  表示区间和方向在同一个半球，flag == -1 表示在两个半球。
            if (range_phi(2) > pi && east == 0) || (range_phi(2) <= pi && east == 1)
                flag =1;
            else 
                flag =-1;
            end
            if range_phi(1) <= pi
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
            %% find_maximum 
            delta_phi_near = abs(phi_near - outer_phi);
            if delta_phi_near ==0
                maximum = dot(x,polar_2_xyz(alpha_near,phi_near));
            else 
                maximum = max(  dot(x,polar_2_xyz(range_alpha(1),phi_near)),...
                                dot(x,polar_2_xyz(range_alpha(2),phi_near)));
            end
            %% find_minimum
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
    else
        for i = 1:N
            east = line_pair.outer_product_belong(i);
            if (range_phi(2) > pi && east == 0) || (range_phi(2) <= pi && east == 1)
                flag =1;
            else 
                flag =-1;
            end
            if range_phi(1) <= pi
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

            %% find_maximum 
            delta_phi_near = abs(phi_near - outer_phi);
            if delta_phi_near ==0
                maximum = dot(x,polar_2_xyz(alpha_near,phi_near));
            elseif delta_phi_near>pi/2
                % maximum = max(  dot(x,polar_2_xyz(range_alpha(1),phi_near)),...
                %                 dot(x,polar_2_xyz(range_alpha(2),phi_near)));
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

            %% find_minimum
            delta_phi_far = abs(phi_far - outer_phi);
            if delta_phi_far <pi/2
                % minimum = min(  dot(x,polar_2_xyz(range_alpha(1),phi_far)),...
                %                 dot(x,polar_2_xyz(range_alpha(2),phi_far)));
            % tangent = tan(outer_alpha)*cos(delta_phi_near);
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
               minimum = dot(x,polar_2_xyz(range_alpha(1),phi_far));
            else
               minimum = dot(x,polar_2_xyz(range_alpha(2),phi_far));
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

