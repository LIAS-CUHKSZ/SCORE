function [alpha, phi,theta] = rot2angle(R)
        axis =  rotmat2vec3d(R);
        theta = norm(axis);

        % 计算旋转角alpha和phi
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


    end