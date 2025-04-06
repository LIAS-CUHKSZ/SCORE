%%%%
% Implementation of the FGO rotation estimator under saturated consensus maximization

%%% Inputs:
% vector_v: N x 3, the direction vector for each matched 3D map line.
% vector_n: N x 3, the normal vector paramater for each 2D image line.
% branch_resolution: scalar, stop bnb when cube length < resolution.
% epsilon: scalar, for Sat-CM formulation.
% sample_resolution: scalar, control resolution for interval analysis.
% verbose_flag: bool, set true for detailed bnb process info.
% id: N x 1, the belonging 2D line id for each matched pair.
% kernel: function, the saturation function.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%
function [R_opt,best_upper_record,best_lower_record,best_lower,num_candidate,time] = Sat_RotFGO(vector_v,vector_n,branch_resolution,epsilon,sample_resolution,verbose_flag,id,kernel)
%%% data pre-process
N = size(vector_n,1);
[outer_product,outer_east,outer_west,  inner_product, normal_east, normal_west,  o_normal_east,  o_normal_west] = data_process(vector_n,vector_v);
vector_normal_east = zeros(N,3);
vector_normal_west = zeros(N,3);
vector_o_normal_east = zeros(N,3);
vector_o_normal_west = zeros(N,3);
vector_outer_west =zeros(N,3);
vector_outer_east= zeros(N,3);
outer_norm =zeros(N,1);
outer_product_belong = zeros(N,1); % which half sphere the vector belongs to  1 for east and 0 for west
for i = 1:N
    vector_normal_east(i,:) = [sin(normal_east(i,1))*cos(normal_east(i,2)),sin(normal_east(i,1))*sin(normal_east(i,2)),cos(normal_east(i,1))];
    vector_normal_west(i,:) = [sin(normal_west(i,1))*cos(normal_west(i,2)),sin(normal_west(i,1))*sin(normal_west(i,2)),cos(normal_west(i,1))];
    vector_o_normal_east(i,:) = [sin(o_normal_east(i,1))*cos(o_normal_east(i,2)),sin(o_normal_east(i,1))*sin(o_normal_east(i,2)),cos(o_normal_east(i,1))];
    vector_o_normal_west(i,:) = [sin(o_normal_west(i,1))*cos(o_normal_west(i,2)),sin(o_normal_west(i,1))*sin(o_normal_west(i,2)),cos(o_normal_west(i,1))];
    outer_norm(i) = norm(outer_product(i,:));
    if outer_product(i,2) >= 0
        outer_product_belong(i) = 1;
        vector_outer_east(i,:) =outer_product(i,:);
        vector_outer_west(i,:) =-outer_product(i,:);
    else
        vector_outer_east(i,:) =-outer_product(i,:);
        vector_outer_west(i,:) = outer_product(i,:);
        outer_product_belong(i) = 0;
    end
end
line_pair.outer_product_belong = outer_product_belong;
line_pair.vector_normal_east = vector_normal_east;
line_pair.vector_normal_west = vector_normal_west;
line_pair.vector_o_normal_east = vector_o_normal_east;
line_pair.vector_o_normal_west = vector_o_normal_west;
line_pair.inner_product = inner_product;
line_pair.outer_product = outer_product;
line_pair.vector_outer_east = vector_outer_east;
line_pair.vector_outer_west = vector_outer_west;
line_pair.normal_east = normal_east;
line_pair.normal_west = normal_west;
line_pair.o_normal_east = o_normal_east;
line_pair.o_normal_west = o_normal_west;
line_pair.vector_n = vector_n;
line_pair.vector_v = vector_v;
line_pair.size = N;
line_pair.outer_east = outer_east;
line_pair.outer_west = outer_west;
line_pair.outer_norm =outer_norm;
%%% start the BnB process
% Initialization
branch=[];
B_east=[0;0;pi;pi]; B_west=[0;pi;pi;2*pi]; % split into two semisphere.
theta_0=zeros(2,1);
[upper_east,lower_east,theta_0(1)]=Sat_Bounds_FGO(line_pair,B_east,epsilon,sample_resolution,id,kernel);    
branch=[branch,[B_east;upper_east;lower_east]];
[upper_west,lower_west,theta_0(2)]=Sat_Bounds_FGO(line_pair,B_west,epsilon,sample_resolution,id,kernel);
branch=[branch,[B_west;upper_west;lower_west]];
best_upper = max(upper_east,upper_west);
ind_upper = 2-(upper_east>upper_west);
best_lower = max(lower_east,lower_west);
ind_lower = 2-(lower_east>lower_west);
theta_best = theta_0(ind_lower);
r_branch=branch(1:4,ind_lower);
u_best=polar_2_xyz(0.5*(r_branch(1)+r_branch(3)),0.5*(r_branch(2)+r_branch(4)));
%%%
next_branch=branch(1:4,ind_upper);
branch(:,ind_upper)=[];
branch(:,branch(5,:)<best_lower)=[];
new_upper=zeros(1,4);
new_lower=zeros(1,4);
new_theta_lower=zeros(1,4);
iter=1;
best_upper_record=best_upper;
best_lower_record=best_lower;
tic
while true
    new_branch=subBranch(next_branch);
    for i=1:4
        [new_upper(i),new_lower(i),new_theta_lower(i)]=Sat_Bounds_FGO(line_pair,new_branch(:,i),epsilon,sample_resolution,id,kernel);
        if(verbose_flag)
            fprintf('Iteration: %d, Branch: [%f, %f, %f, %f], Upper: %f, Lower: %f\n', iter, new_branch(:,i), new_upper(i), new_lower(i));
        end
    end
    branch=[branch,[new_branch;new_upper;new_lower]];
    % branch_t=branch';
    for i=1:4
        if  floor(new_lower(i))>floor(best_lower)
            best_lower=new_lower(i);
            r_branch=new_branch(1:4,i);
            u_best=polar_2_xyz(0.5*(new_branch(1,i)+new_branch(3,i)) , 0.5*(new_branch(2,i)+new_branch(4,i)) );
            theta_best = new_theta_lower(i);
        elseif floor(new_lower(i))==floor(best_lower)
            u_new=polar_2_xyz(0.5*(new_branch(1,i)+new_branch(3,i)),0.5*(new_branch(2,i)+new_branch(4,i)));
            if max(u_new'*u_best)<cosd(3) % the new axis is not proximate to cur axises
                r_branch=[r_branch,new_branch(1:4,i)];
                theta_best = [theta_best,new_theta_lower(i)];
                u_best = [u_best,u_new];
            end
        else
        end
    end
    % best_upper_record=[best_upper_record;best_upper];
    % best_lower_record=[best_lower_record;best_lower];
    [best_upper,ind_upper]=max(branch(5,:));
    next_branch=branch(1:4,ind_upper);
    branch(:,ind_upper)=[];
    branch(:,branch(5,:)<best_lower)=[];
    if (  (new_branch(3,1) - new_branch(1,1)) < branch_resolution )
        break;
    end
    iter=iter+1;
end
%% stop branching
num_candidate=size(r_branch,2);
R_opt=[];
for i=1:num_candidate
    R_opt = [R_opt;rotation_from_axis_angle(u_best(:,i), theta_best(i))];
end
time=toc;
end

