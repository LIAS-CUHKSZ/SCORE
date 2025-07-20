function [theta] = angular_distance(R_hat,R)
%ANGULAR_DISTANCE 
%out put the angular distance between estimated and true rotation (unit: degrees)
theta=acosd(0.5*( trace(R_hat'*R)-1 ));
end

