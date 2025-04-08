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
line_pair_data = data_process(vector_n,vector_v);
%%%%%%%%%%%%%%%%%%%%% Acclerated BnB %%%%%%%%%%%%%%%%%%%%%
tic
%%% Initialize the BnB process
% calculate bounds for east and west semispheres.
branch=[];
B_east=[0;0;pi;pi]; B_west=[0;pi;pi;2*pi];
theta_0=zeros(2,1);
[upper_east,lower_east,theta_0(1)]=Sat_Bounds_FGO(line_pair_data,B_east,epsilon,sample_resolution,id,kernel);    
branch=[branch,[B_east;upper_east;lower_east]];
[upper_west,lower_west,theta_0(2)]=Sat_Bounds_FGO(line_pair_data,B_west,epsilon,sample_resolution,id,kernel);
branch=[branch,[B_west;upper_west;lower_west]];
% record the current best estimate according to the lower bounds
best_lower = max(lower_east,lower_west);
ind_lower = 2-(lower_east>lower_west);
theta_best = theta_0(ind_lower);
r_branch=branch(1:4,ind_lower);
u_best=polar_2_xyz(0.5*(r_branch(1)+r_branch(3)),0.5*(r_branch(2)+r_branch(4)));
% select the next exploring branch according to the upper bounds
best_upper = max(upper_east,upper_west);
ind_upper = 2-(upper_east>upper_west);
next_branch=branch(1:4,ind_upper);
branch(:,ind_upper)=[];
% record bounds history
best_upper_record=best_upper; best_lower_record=best_lower;
%%% start BnB
new_upper=zeros(1,4); new_lower=zeros(1,4); new_theta_lower=zeros(1,4);
iter=1;
while true
    new_branch=subBranch(next_branch);
    for i=1:4
        [new_upper(i),new_lower(i),new_theta_lower(i)]=Sat_Bounds_FGO(line_pair_data,new_branch(:,i),epsilon,sample_resolution,id,kernel);
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
%%% output
num_candidate=size(r_branch,2);
R_opt=[];
for i=1:num_candidate
    R_opt = [R_opt;rotation_from_axis_angle(u_best(:,i), theta_best(i))];
end
time=toc;
end


function out=subBranch(branch)
% divide the input 2D cube into four subcubes
a=branch(1:2);
b=branch(3:end);
c=0.5*(a+b);
M=[a,c,b];
out=zeros(4,4);
for i=1:4
    out(1,i)= M(1,bitget(i,1)+1);
    out(2,i)= M(2,bitget(i,2)+1);
    out(3,i)= M(1,bitget(i,1)+2);
    out(4,i)= M(2,bitget(i,2)+2);
end

end