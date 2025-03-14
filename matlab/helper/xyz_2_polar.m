function [alpha,phi] = xyz_2_polar(axis)
%XYZ_2_POLAR 此处显示有关此函数的摘要
%   此处显示详细说明
    
    length = norm(axis);
    if length ==0
        alpha =0;
        phi =0;
        return;
    else 
        axis = axis/length;
    end
    if axis(1)==0 && axis(2)==0
        phi = 0;
        alpha = acos(axis(3));
    elseif axis(1)==0
        phi = pi/2;
        alpha = acos(axis(3));
    else
        phi = atan2(axis(2), axis(1));
        
        alpha = atan2( sqrt(axis(1)^2 + axis(2)^2), axis(3));
    end
    if phi<0
        phi = phi+2*pi;
    end
end

