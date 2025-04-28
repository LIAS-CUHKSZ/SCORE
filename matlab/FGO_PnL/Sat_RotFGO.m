%%%%
% Implementation of the FGO rotation estimator under saturated consensus maximization

%%% Inputs:
% vector_n: N x 3, the normal vector paramater for each 2D image line.
% vector_v: N x 3, the direction vector for each matched 3D map line.
% id: N x 1, the belonging 2D line id for each matched pair.
% kernel_buffer: store weights given by the selected saturation function.
% branch_reso: scalar, stop bnb when cube length < resolution.
% epsilon_r: scalar, for Sat-CM formulation.
% sample_reso: scalar, control resolution for interval analysis.
% verbose_flag: bool, set true for detailed bnb process info.
% mex_flag: bool, set true to use mex code for acceleration.



%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [R_opt,best_lower,num_candidate,time,upper_record,lower_record] = Sat_RotFGO(vector_n,vector_v,id,kernel_buffer,branch_reso,epsilon_r,sample_reso,verbose_flag,mex_flag)
%%% paramaters for handling unbiguity of the global optimum
rounding_digit = 9;
proximity_thres = cosd(5);
%%% data pre-process
line_pair_data = data_process(vector_n,vector_v);
%%%%%%%%%%%%%%%%%%%%% Acclerated BnB %%%%%%%%%%%%%%%%%%%%%
tic
%%% Initialize the BnB process
% calculate bounds for east and west semispheres.
branch=[];
B_east=[0;0;pi;pi]; B_west=[0;pi;pi;2*pi];
theta_0=zeros(2,1);
if mex_flag
    [upper_east,lower_east,theta_0(1)]=Sat_Bounds_FGO_mex(line_pair_data,B_east,epsilon_r,sample_reso,id,kernel_buffer);    
    [upper_west,lower_west,theta_0(2)]=Sat_Bounds_FGO_mex(line_pair_data,B_west,epsilon_r,sample_reso,id,kernel_buffer);
else
    [upper_east,lower_east,theta_0(1)]=Sat_Bounds_FGO(line_pair_data,B_east,epsilon_r,sample_reso,id,kernel_buffer);    
    [upper_west,lower_west,theta_0(2)]=Sat_Bounds_FGO(line_pair_data,B_west,epsilon_r,sample_reso,id,kernel_buffer);
end
branch=[branch,[B_east;upper_east;lower_east]];
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
upper_record=best_upper; lower_record=best_lower;
%%% start BnB
new_upper=zeros(1,4); new_lower=zeros(1,4); new_theta_lower=zeros(1,4);
iter=1;
while true
    new_branch=subBranch(next_branch);
    for i=1:4
        if mex_flag
            [new_upper(i),new_lower(i),new_theta_lower(i)]=Sat_Bounds_FGO_mex(line_pair_data,new_branch(:,i),epsilon_r,sample_reso,id,kernel_buffer);
        else
            [new_upper(i),new_lower(i),new_theta_lower(i)]=Sat_Bounds_FGO(line_pair_data,new_branch(:,i),epsilon_r,sample_reso,id,kernel_buffer);
        end
        if(verbose_flag)
            fprintf('Iteration: %d, Branch: [%f, %f, %f, %f], Upper: %f, Lower: %f\n', iter, new_branch(:,i), new_upper(i), new_lower(i));
        end
    end
    new_upper=round(new_upper,rounding_digit);
    new_lower=round(new_lower,rounding_digit);
    branch=[branch,[new_branch;new_upper;new_lower]];
    % branch_t=branch';
    for i=1:4
        if  new_lower(i)>best_lower
            best_lower=new_lower(i);
            r_branch=new_branch(1:4,i);
            u_best=polar_2_xyz(0.5*(new_branch(1,i)+new_branch(3,i)) , 0.5*(new_branch(2,i)+new_branch(4,i)) );
            theta_best = new_theta_lower(i);
        elseif new_lower(i)==best_lower
            u_new=polar_2_xyz(0.5*(new_branch(1,i)+new_branch(3,i)),0.5*(new_branch(2,i)+new_branch(4,i)));
            if max(u_new'*u_best)<proximity_thres % the new axis is not proximate to cur axises
                r_branch=[r_branch,new_branch(1:4,i)];
                theta_best = [theta_best,new_theta_lower(i)];
                u_best = [u_best,u_new];
            end
        else
        end
    end
    upper_record=[upper_record;best_upper];
    lower_record=[lower_record;best_lower];
    [best_upper,ind_upper]=max(branch(5,:));
    next_branch=branch(1:4,ind_upper);
    branch(:,ind_upper)=[];
    branch(:,branch(5,:)<best_lower)=[];
    if (  (new_branch(3,1) - new_branch(1,1)) < branch_reso )
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