%%%
% confine the search space of rotation axis around the input axis
% alpha: [0,pi]
% phi  : [0,2*pi]
% side_length: pi, pi/2, pi/4, ...
% delta: sclar, define the ambiguous region

%%% Author:  Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
function branch = confine_sphere(alpha,phi,side_length,delta)
eps = 10^(-14);
branch=[];
%
branch_ns_list = [];
k_alpha = floor(alpha/side_length);
if alpha-k_alpha*side_length <= delta && k_alpha>0
   branch_ns_list = [[(k_alpha-1)*side_length;k_alpha*side_length],[k_alpha*side_length;(k_alpha+1)*side_length]];
elseif (k_alpha+1)*side_length-alpha <= delta && (k_alpha+1)*side_length+eps < pi
   branch_ns_list = [[k_alpha*side_length;(k_alpha+1)*side_length],[(k_alpha+1)*side_length;(k_alpha+2)*side_length]];
else
   branch_ns_list = [k_alpha*side_length;(k_alpha+1)*side_length];
end
%
branch_we_list = [];
k_phi = floor(phi/side_length);
if phi - k_phi*side_length <=delta
   branch_we_list = [[(k_phi-1)*side_length;k_phi*side_length],[k_phi*side_length;(k_phi+1)*side_length]];
elseif (k_phi+1)*side_length-phi <= delta
   branch_we_list = [[k_phi*side_length;(k_phi+1)*side_length],[(k_phi+1)*side_length;(k_phi+2)*side_length]];
else
   branch_we_list= [k_phi*side_length;(k_phi+1)*side_length];
end

% push branch
for i = 1:size(branch_we_list,2)
    phi_l = branch_we_list(1,i);
    phi_u = branch_we_list(2,i);
    if phi_l<0
        phi_l = 2*pi-side_length;
        phi_u = 2*pi;
    end
    if phi_u>2*pi
       phi_l = 0;
       phi_u = side_length;
    end
    for j = 1:size(branch_ns_list,2)
        alpha_l = branch_ns_list(1,j);
        alpha_u = branch_ns_list(2,j);
        branch = [branch,[alpha_l;phi_l;alpha_u;phi_u]];
    end
end

end