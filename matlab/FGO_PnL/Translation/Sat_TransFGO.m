%%%%
% Implementation of the FGO translation estimator under saturated consensus maximization

%%% Inputs:
% pert_rot_n_2D:    L  x 3, the rotated and perturbed normal vector paramater for each 2D image line.
% endpoints_3D:     2L x 3, the endpoints of 3D lines
% ids:              L x 1, the belonging 2D line id for each matched pair.
% sat_buff:         M x N, store weights given by the selected saturation function.
% space_size:       3 x 1, the space bounding box
% branch_reso:      scalar, stop bnb when cube length < resolution.
% epsilon_t:        scalar, error tolerance
% prox_thres:       scalar, used for clustering proximate stabbers

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [t_best,best_lower,num_candidate,time,upper_record,lower_record] = Sat_TransFGO(pert_rot_n_2D,endpoints_3D,ids,sat_buff,space_size,branch_reso,epsilon_t,prox_thres)
mex_flag=1;     %    bool, set true to use mex code for acceleration.
if mex_flag
    UB_fh = @Sat_Trans_U_mex;
    LB_fh = @Sat_Trans_L_mex;
else
    UB_fh = @Sat_Trans_U;
    LB_fh = @Sat_Trans_L;
end
% we use eps to avoid numerical error when comparing two double variable
eps = min(10^(-8),min(sat_buff(sat_buff>0))/2);

% --- 1. Initialization ---
tic
% reduce dimension (x) and divide (y,z) into cubes of 1 meters
ceil_y = ceil(space_size(2)); ceil_z = ceil(space_size(3));
branch = zeros(5,ceil_y*ceil_z);  % each column: y_min, z_min, y_max, z_max, upper bound
upper_record=[]; lower_record=[]; % record optimal bounds history
p_3D = endpoints_3D(1:2:end,:);

% bound the initial cubes with side length 1 meter
best_lower = -1; 
for i=1:ceil_y
    for j=1:ceil_z
        idx = (i-1)*ceil_z+j;
        br_ = [i-1;j-1;i;j];
        % calculate the upper bound
        u_ = UB_fh(pert_rot_n_2D,p_3D,ids,epsilon_t,br_,space_size,sat_buff);
        % push into container
        branch(:,idx)=[br_;u_];
    end
end

% --- 2. Start the Acclerated BnB process ---
best_branch = [];
iter = 1;
while ~isempty(branch)
    % pop out the branch with highest upper bound
    [best_upper,max_idx] = max(branch(5,:));
    popped_branch = branch(:,max_idx);
    branch(:,max_idx)=[];
    [lower_bound,~] = LB_fh(pert_rot_n_2D,p_3D,ids,epsilon_t,popped_branch(1:4),space_size,sat_buff,prox_thres);
    
    % update best lower bound and optimal theta
    if lower_bound>best_lower+eps  % lower_bound>best_lower
       best_lower = lower_bound;
       best_branch = popped_branch(1:4);
       
    elseif lower_bound+eps > best_lower % lower_bound == best_lower
       best_branch = [best_branch,popped_branch(1:4)]; %append the list

    else
        
    end

    % record the history of best lower/upper bounds
    lower_record=[lower_record;best_lower];
    upper_record=[upper_record;best_upper];
    iter = iter+1;

    % prune branches according to the updated lower bound
    branch(:,(branch(5,:)+eps)<best_lower)=[]; 
    
    % terminate further branching if reaching resolution
    if popped_branch(3,:)-popped_branch(1,:)+10^(-8)<branch_reso 
        continue;
    end

    % terminate further branching if 
    % U/L bounds meet, and current cube attains the best lower bound.
    if (best_upper < best_lower + eps) && (lower_bound+eps > best_lower) 
       % best_upper == best_lower && lower_bound == best_lower
       continue;
    else % divide the branch into four
        new_branch=subBranch(popped_branch(1:4)); 
        % evaluate the upper bounds for each sub branch
        for i = 1:4
            new_upper = UB_fh(pert_rot_n_2D,p_3D,ids,epsilon_t,new_branch(:,i),space_size,sat_buff);
            branch=[branch,[new_branch(:,i);new_upper]];
        end
    end

    % record bounds history
    best_upper  = max(branch(5,:));
    upper_record=[upper_record;best_upper];
    lower_record=[lower_record;best_lower];
end

% --- 3. Output ---
time=toc;
t_best=[];
num_candidate=0;
for i = 1 : size(best_branch,2)
    this_branch = best_branch(:,i);
    yz_sampled = [this_branch(1)+this_branch(3);this_branch(2)+this_branch(4)]/2;
    [~,x_opt] = LB_fh(pert_rot_n_2D,p_3D,ids,epsilon_t,this_branch,space_size,sat_buff,prox_thres);
    x_opt = cluster_stabber(x_opt,prox_thres);
    for j = 1:length(x_opt)
        append_t = [x_opt(j);yz_sampled];
        t_best=[t_best,append_t];
    end
    num_candidate = num_candidate + length(x_opt);
end
end
