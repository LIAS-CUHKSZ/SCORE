% transform a unit-length verctor to polar coordinates
function [alpha,phi] = xyz_2_polar(axis)
    length = norm(axis);
    if length ==0
        alpha =0;
        phi =0;
        return;
    else 
        axis = axis/length;   %normalization
    end
    if axis(1)==0 && axis(2)==0
        phi = 0;
        alpha = acos(axis(3));
    elseif axis(1)==0
        pi_2 = pi/2;
        if axis(2) <0
            phi = pi+pi_2;
        else
            phi =pi_2;
        end
        alpha = acos(axis(3));
    else
        phi = atan2(axis(2), axis(1));
        
        alpha = atan2( sqrt(axis(1)^2 + axis(2)^2), axis(3));
    end
    if phi<0
        phi = phi+2*pi;
    end
end

