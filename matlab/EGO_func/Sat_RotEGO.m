%%%%
% wrap EGO rotation estimator with the same BnB framework as our method.
%%% Inputs:
% vector_n: N x 3, the normal vector paramater for each 2D image line.
% vector_v: N x 3, the direction vector for each matched 3D map line.
% id: N x 1, the belonging 2D line id for each matched pair.
% kernel_buffer: store weights given by the selected saturation function.
% branch_reso: scalar, stop bnb when cube length < resolution.
% epsilon_r: scalar, for Sat-CM formulation.
% sample_reso: scalar, control resolution for interval analysis.
% prox_thres:  double, the candidates are not proximate to each other.
% verbose_flag: bool, set true for detailed bnb process info.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.2
%%% License: MIT
%%%%

function [R_opt,best_lower,num_candidate,time,upper_record,lower_record] = Sat_RotEGO(vector_n,vector_v,ids,kernel_buff,branch_reso,epsilon_r,prox_thres,verbose_flag)
%%% data pre-process
line_pair_data = data_process(vector_n,vector_v);
%%%%%%%%%%%%%%%%%%%%% Acclerated BnB %%%%%%%%%%%%%%%%%%%%%
tic
%%% Initialize the BnB process
% calculate bounds for east and west semispheres.
branch=[];
B_east=[0;0;pi;pi]; B_west=[0;pi;pi;2*pi];
[upper_east,lower_east,theta_east]=get_bounds_sat(vector_v',vector_n',B_east,epsilon_r,ids,kernel_buff);    
[upper_west,lower_west,theta_west]=get_bounds_sat(vector_v',vector_n',B_west,epsilon_r,ids,kernel_buff);    
branch=[branch,[B_east;upper_east;lower_east]];
branch=[branch,[B_west;upper_west;lower_west]];
% record the current best estimate according to the lower bounds
best_lower = max(lower_east,lower_west);
if lower_east>=lower_west
   theta_best = cluster_stabber(theta_east,prox_thres);
   u_best=polar_2_xyz(0.5*(B_east(1)+B_east(3)),0.5*(B_east(2)+B_east(4)));
else
   theta_best = cluster_stabber(theta_west,prox_thres);
   u_best=polar_2_xyz(0.5*(B_west(1)+B_west(3)),0.5*(B_west(2)+B_west(4)));
end
u_best=repmat(u_best,1,length(theta_best)); % it is possible that multiple optimal stabbers returned by Sat-IS 
% select the next exploring branch according to the upper bounds
best_upper = max(upper_east,upper_west);
idx_upper = 2-(upper_east>upper_west);
next_branch=branch(1:4,idx_upper);
branch(:,idx_upper)=[];
% record bounds history
upper_record=best_upper; lower_record=best_lower;
%%% start BnB
new_upper=zeros(1,4); new_lower=zeros(1,4); 
new_theta_lower=cell(1,4); % it is possible that multiple optimal stabbers returned by Sat-IS
iter=1;
while true
    new_branch=subBranch(next_branch);
    for i=1:4
        [new_upper(i),new_lower(i),new_theta_lower{i}]=get_bounds_sat(line_pair_data.vector_v',line_pair_data.vector_n',new_branch(:,i),epsilon_r,ids,kernel_buff);
        if(verbose_flag)
            fprintf('Iteration: %d, Branch: [%f, %f, %f, %f], Upper: %f, Lower: %f\n', iter, new_branch(:,i), new_upper(i), new_lower(i));
        end
    end
    branch=[branch,[new_branch;new_upper;new_lower]];
    % branch_t=branch';
    for i=1:4
        if  new_lower(i)>best_lower
            best_lower=new_lower(i);
            u_best=polar_2_xyz(0.5*(new_branch(1,i)+new_branch(3,i)) , 0.5*(new_branch(2,i)+new_branch(4,i)) );
            theta_best = cluster_stabber(new_theta_lower{i},prox_thres);
            u_best=repmat(u_best,1,length(theta_best));
            continue;
        end
        if new_lower(i)==best_lower
            u_new=polar_2_xyz(0.5*(new_branch(1,i)+new_branch(3,i)),0.5*(new_branch(2,i)+new_branch(4,i)));
            new_best_theta = cluster_stabber(new_theta_lower{i},prox_thres);
            % when achieve the same best score, store the new rotation if
            % the difference with currently stored rotation is salient.
            for j=1:length(new_best_theta)
                R_new=rotvec2mat3d(u_new*new_best_theta(j));
                prox_flag = evaluate_proximity(theta_best,u_best,R_new,prox_thres);
                if ~prox_flag
                    theta_best = [theta_best,new_best_theta(j)];
                    u_best = [u_best,u_new];
                end
            end
        end
    end
    upper_record=[upper_record;best_upper];
    lower_record=[lower_record;best_lower];
    best_upper = max(branch(5,:));
    idx_upper = find(branch(5,:)==best_upper);
    branch_size=branch(3,idx_upper)-branch(1,idx_upper);
    [~,temp_idx]=max(branch_size);
    idx_upper=idx_upper(temp_idx);
    next_branch=branch(1:4,idx_upper);
    branch(:,idx_upper)=[];
    branch(:,branch(5,:)<best_lower)=[];
    if (  (next_branch(3,1) - next_branch(1,1)) < branch_reso )
        break;
    end
    iter=iter+1;
end
%%% output
num_candidate=size(u_best,2);
R_opt=[];
for i=1:num_candidate
    R_opt = [R_opt;rotvec2mat3d(u_best(:,i)*theta_best(i))];
end
time=toc;
end

%%%%%%%%%%%%%%%%%%%%% subfunctions %%%%%%%%%%%%%%%%%%%%%%
function stabber_clustered=cluster_stabber(stabbers,thres)
if isscalar(stabbers)
    stabber_clustered=stabbers;
    return
end
stabber_clustered=[];
N = length(stabbers);
stabber_buffer=stabbers(1);
for n=2:N
    new_stabber=stabbers(n);
    if new_stabber-stabber_buffer(1)>thres
        temp_idx = 1:length(stabber_buffer)-1+mod(length(stabber_buffer),2);
        stabber_clustered=[stabber_clustered,median(stabber_buffer(temp_idx))];
        stabber_buffer=new_stabber;
    else
        stabber_buffer=[stabber_buffer,new_stabber];
    end
end
    temp_idx = 1:length(stabber_buffer)-1+mod(length(stabber_buffer),2);
    stabber_clustered=[stabber_clustered,median(stabber_buffer(temp_idx))];
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
%
function prox_flag = evaluate_proximity(theta_best,u_best,R_new,prox_thres)
    prox_flag=0;
    for k=1:length(theta_best)
        R_old=rotvec2mat3d(u_best(:,k)*theta_best(k));
        ad = angular_distance(R_new,R_old);
        if ad*pi/180<=prox_thres
            prox_flag=1;
            break;
        end
    end
end