function [axis] = polar_2_xyz(alpha,phi)
%POLAR_2_XYZ 此处显示有关此函数的摘要
%   此处显示详细说明
    % axis = zeros(3,1);
    % a_s = sin(alpha);
    % a_c = cos(alpha);
    % p_s = sin(phi);
    % p_c = cos(phi);
    % axis(1)= a_s*p_c;
    % axis(2)= a_s*p_s;
    % axis(3)= a_c;
    axis = zeros(3,1);
    a_s = sin(alpha);
    axis(1)= a_s*cos(phi);
    axis(2)= a_s*sin(phi);
    axis(3)= cos(alpha);
end

