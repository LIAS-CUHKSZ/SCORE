% transform rotation matrix to polar coodinates (of rotation axis) and
% rotation angle (theta)
function [alpha, phi,theta] = rot2angle(R)
        rot_vec =  rotmat2vec3d(R);
        theta = norm(rot_vec);
        axis = rot_vec / theta;
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
        if phi <0 % [-pi,pi]-->[0,2pi]
            phi = phi + 2*pi;
        end
    end