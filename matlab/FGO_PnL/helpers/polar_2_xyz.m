% from polar coodinates to xyz 
function [axis] = polar_2_xyz(alpha,phi)
    axis = zeros(3,1);
    a_s = sin(alpha);
    axis(1)= a_s*cos(phi);
    axis(2)= a_s*sin(phi);
    axis(3)= cos(alpha);
end

