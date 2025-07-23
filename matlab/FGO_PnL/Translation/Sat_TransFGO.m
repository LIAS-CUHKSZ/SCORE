%%%%
% Implementation of the FGO translation estimator under saturated consensus maximization

%%% Inputs:
% pert_rot_n_2D:    L  x 3, the rotated and perturbed normal vector paramater for each 2D image line.
% endpoints_3D:     2L x 3, the endpoints of 3D lines
% ids:              L x 1, the belonging 2D line id for each matched pair.
% kernel_buff:      M x N, store weights given by the selected saturation function.
% space_size:       3 x 1, the space bounding box
% branch_reso:      scalar, stop bnb when cube length < resolution.
% epsilon_t:        scalar, error tolerance
% prox_thres:       double, the candidates are not proximate to each other.



%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [t_best,best_lower,num_candidate,time,upper_record,lower_record] = Sat_TransFGO(pert_rot_n_2D,endpoints_3D,ids,kernel_buff,space_size,branch_reso,epsilon_t,prox_thres)
verbose_flag=0; %    bool, set true for detailed bnb process info.
mex_flag=1;     %    bool, set true to use mex code for acceleration.
if mex_flag
   Bound_fh = @Sat_Bounds_trans_mex; % mex_compiled bound function
else
   Bound_fh = @Sat_Bounds_trans; % original bound function
end

% --- 1. Initialization ---
tic
% reduce dimension (x) and divide (y,z) into cubes of 1 meters
ceil_y = ceil(space_size(2)); ceil_z = ceil(space_size(3));
branch = zeros(6,ceil_y*ceil_z);  % each column: y_min, z_min, y_max, z_max, lower bound, upper bound

% bound the initial cubes with side length 1 meter
best_lower = -1; t_best=zeros(3,1);
for i=1:ceil_y
    for j=1:ceil_z
        idx = (i-1)*ceil_z+j;
        br_ = [i-1;j-1;i;j];
        [u_,l_,t_sample] = Bound_fh(pert_rot_n_2D,endpoints_3D(1:2:end,:),ids,epsilon_t,br_,space_size,kernel_buff,prox_thres);
        branch(:,idx)=[br_;u_;l_];
        if l_ > best_lower
            t_best = t_sample;
            best_lower = l_;
        elseif l_== best_lower
            t_best = [t_best,t_sample];
        else
        end
    end
end

% --- 2. BnB Start ---
best_upper  = max(branch(5,:));
upper_record=best_upper; lower_record=best_lower; 
iter=1;
while true
    % terminating condition 0
    if best_upper <= best_lower
        break;
    end

    % prune branches
    branch(:,branch(5,:)<best_lower)=[];

    % select the next exploring branch according to the upper bounds
    idx_upper   = find(branch(5,:)==best_upper);
    branch_size = branch(3,idx_upper)-branch(1,idx_upper);
    [~,temp_idx]= max(branch_size); idx_upper=idx_upper(temp_idx); 
    % choose the one with largest upper bound and largest width
    next_branch=branch(1:4,idx_upper); 
    branch(:,idx_upper)=[];
    if isempty(next_branch)
        next_branch
    end
    % terminating condition 1
    if (  (next_branch(3,1) - next_branch(1,1)) < branch_reso )
        break;
    end

    % divide and bound the new branches
    new_branch=subBranch(next_branch);
    new_upper = zeros(1,4); new_lower = zeros(1,4); new_t_sample = cell(4,1);
    for i=1:4
        [new_upper(i),new_lower(i),new_t_sample{i}]=Bound_fh(pert_rot_n_2D,endpoints_3D(1:2:end,:),ids,epsilon_t,new_branch(:,i),space_size,kernel_buff,prox_thres);
        if verbose_flag
            fprintf('Iteration: %d, Branch: [%f, %f, %f, %f], Upper: %f, Lower: %f\n', iter, new_branch(:,i), new_upper(i), new_lower(i));
        end
    end
    branch=[branch,[new_branch;new_upper;new_lower]];
    
    % update the best lower bound and candidate
    if max(new_lower)==0
        continue
    end
    for i=1:4
        if  new_lower(i)>best_lower
            best_lower=new_lower(i);
            x_stabber = new_t_sample{i}(1,:); yz_sample = new_t_sample{i}(2:3,1);
            x_cluster = cluster_stabber(x_stabber,prox_thres);
            x_num = length(x_cluster); clustered_t_sample = zeros(3,x_num);
            clustered_t_sample(2:3,:) = repmat(yz_sample,[1,x_num]);
            clustered_t_sample(1,:) = x_cluster;
            t_best = clustered_t_sample;
            continue;
        end
        if new_lower(i)==best_lower
            x_stabber = new_t_sample{i}(1,:); yz_sample = new_t_sample{i}(2:3,1);
            x_cluster = cluster_stabber(x_stabber,prox_thres);
            x_num = length(x_cluster); clustered_t_sample = zeros(3,x_num);
            clustered_t_sample(2:3,:) = repmat(yz_sample,[1,x_num]);
            clustered_t_sample(1,:) = x_cluster;
            %append the best candiate lists
            t_best = [t_best, clustered_t_sample]; 
        end
    end

    % record bounds history
    best_upper  = max(branch(5,:));
    upper_record=[upper_record;best_upper];
    lower_record=[lower_record;best_lower];

    % terminating condition 2
    iter=iter+1;
    if iter > 500
        break
    end
end

% --- 3. BnB Start ---
num_candidate=size(t_best,2);
time=toc;
end
