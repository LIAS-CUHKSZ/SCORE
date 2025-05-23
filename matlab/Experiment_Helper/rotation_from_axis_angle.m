function R = rotation_from_axis_angle(axis, angle)
    
    ax = axis(1, 1);
    ay = axis(2, 1);
    az = axis(3, 1);
    
    c = cos(angle);
    s = sin(angle);
    
    R = zeros(3, 3);
    R(1, :) = [c+(1-c)*ax^2, (1-c)*ax*ay-s*az, (1-c)*ax*az+s*ay];
    R(2, :) = [(1-c)*ax*ay+s*az, c+(1-c)*ay^2, (1-c)*ay*az-s*ax];
    R(3, :) = [(1-c)*ax*az-s*ay, (1-c)*ay*az+s*ax, c+(1-c)*az^2];
    
end