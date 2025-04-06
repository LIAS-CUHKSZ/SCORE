%%%%
%   in order to solve interval for theta, simultaneously satisfying 
%   A_1*sin(theta + phi_1)+ const_1 >=-epsilon
%   and
%   A_2*sin(theta + phi_2)+ const_2 <= epsilon

%%% Inputs:
% A_1, A_2:             Nx1
% phi_1, phi_2:         Nx1
% const_1, const_2:     Nx1
% epsilon:              scalar

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [interval] = upper_interval(A_1,phi_1,const_1,A_2,phi_2,const_2,epsilon)
% A_1*sin(theta + phi_1)+ const_1 >=-epsilon
interval = [];
upper_interval = [];
c_lo = -const_1 -epsilon;
if A_1 < c_lo
    return;
elseif c_lo>=0
    x_l = asin(c_lo/A_1);
    if phi_1<=pi - x_l
        upper_interval = [max(0,x_l-phi_1);pi-x_l-phi_1];
    elseif phi_1>=pi+x_l
        upper_interval = [2*pi+x_l - phi_1;min(pi,3*pi-x_l-phi_1)]; 
    else
        upper_interval = [];
        return;
    end
elseif c_lo>=-A_1
    x = asin(c_lo/A_1);
    x_l = pi - x;
    x_r = 2*pi+x;
    if phi_1<= x_r-pi
        upper_interval = [0;min(x_l-phi_1,pi)];
    elseif phi_1>=x_l
        upper_interval = [max(0,x_r-phi_1);pi];
    else
        upper_interval = [ 0; x_l-phi_1;  x_r-phi_1;    pi];
    end
else
    upper_interval = [0;pi];
end
%   A_2*sin(theta + phi_2)+ const_2 <= epsilon
lower_interval = [];
c_up =  epsilon-const_2;

if A_2 <= c_up
    lower_interval = [0;pi];
elseif c_up>=0
    x_l = asin(c_up/A_2);
    if phi_2 <= x_l
        lower_interval = [0 ; x_l-phi_2; pi-x_l-phi_2; pi];
    elseif phi_2<=2*pi-x_l 
        % lower_interval = [ max(0,x_l-phi_2);min(pi,2*pi+x_l-phi_2)];
        lower_interval = [ max(0,pi-x_l-phi_2);min(pi,2*pi+x_l-phi_2)];

    else 
        lower_interval = [0; 2*pi+x_l-phi_2; 3*pi-x_l-phi_2; pi];
    end
elseif c_up>=-A_2
    x = asin(c_up/A_2);
    x_l = pi - x;
    x_r = 2*pi+x;
    if phi_2<= -x || phi_2>=x_r
        return;
    else
        lower_interval = [max(0,x_l-phi_2);min(pi,x_r-phi_2)];
    end
else
    return;
end
num_up = length(upper_interval)/2;
num_low = length(lower_interval)/2;
interval = [];
for i=1:num_up
    for j=1:num_low
        interval = [interval;...
                   intersect_interval(upper_interval(2*i-1:2*i,1), lower_interval(2*j-1:2*j,1)) 
                   ];
    end
end
end