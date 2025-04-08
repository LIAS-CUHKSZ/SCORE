function [normal_east,normal_west,o_normal_east,o_normal_west] = normal(v1,v2)
    mid = (v1+v2)/2;
    if(norm(mid)<1e-4)
        normal_east = [0,0];
        normal_west = [0,0];
        [alpha_v1, phi_v1] = xyz_2_polar(v1);
        if(phi_v1>pi)
            o_normal_east = [pi-alpha_v1, phi_v1-pi];
            o_normal_west = [alpha_v1, phi_v1];

        else
            o_normal_east = [alpha_v1, phi_v1];
            o_normal_west = [pi-alpha_v1, phi_v1+pi];
        end

        return;
    end
    mid = mid/norm(mid);
    n =cross(v1,v2);
    n = n/norm(n);
    orthogonal = cross(mid,n);
    orthogonal = orthogonal/norm(orthogonal);
    [alpha_mid, phi_mid] = xyz_2_polar(mid);
    [alpha_orthogonal, phi_orthogonal] = xyz_2_polar(orthogonal);
    if(phi_mid>pi)
        normal_east = [pi-alpha_mid, phi_mid-pi];
        normal_west = [alpha_mid, phi_mid];
    else
        normal_east = [alpha_mid, phi_mid];
        normal_west = [pi-alpha_mid, phi_mid+pi];
    end
    if(phi_orthogonal>pi)
        o_normal_east = [pi-alpha_orthogonal, phi_orthogonal-pi];
        o_normal_west = [alpha_orthogonal, phi_orthogonal];
    else
        o_normal_east = [alpha_orthogonal, phi_orthogonal];
        o_normal_west = [pi-alpha_orthogonal, phi_orthogonal+pi];
    end
end

