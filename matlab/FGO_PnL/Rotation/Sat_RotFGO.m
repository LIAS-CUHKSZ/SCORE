%%%%
% Implementation of the FGO rotation estimator under saturated consensus maximization

%%% Inputs:
% vector_n:         N x 3,  the normal vector paramater of 2D image line for each association.
% vector_v:         N x 3,  the direction vector of 3D map line for each association.
% ids:              N x 1,  the belonging 2D line id for each association.
% sat_buffer:       matrix, stored weights given by the saturation function.
% branch_reso:      scalar, stop bnb when cube length < resolution.
% epsilon_r:        scalar, error tolerance.
% sample_reso:      scalar, control resolution for interval analysis.
% prox_thres:       scalar, used for clustering proximate stabbers
% initial_branch:   M x 4, searched space pruned with prior knowledge
%                   each column [alpha_l;phi_l;alpha_u;phi_u];

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [R_opt,best_lower,num_candidate,time,upper_record,lower_record] = Sat_RotFGO(vector_n,vector_v,ids,sat_buff,branch_reso,epsilon_r,sample_reso,prox_thres, initial_branch)

% --- 0. settings ---
mex_flag = 1; %set true to use mex code for acceleration.
% choose the function handler according to mex_flag
if mex_flag
    UB_fh = @Sat_Rot_U_mex;
    LB_fh = @Sat_Rot_L_mex;
else
    UB_fh = @Sat_Rot_U;
    LB_fh = @Sat_Rot_L;
end

% we use eps to avoid numerical error when comparing two double variable
eps = min(10^(-8),min(sat_buff(sat_buff>0))/2);

% --- 1. preprocess ---
line_pair_data = data_process(vector_n,vector_v); % pre-process data

% --- 2. Initialize the Acclerated BnB process ---
tic
best_lower = -1; 
branch=[];
upper_record=[]; lower_record=[]; % record bounds history
for i = 1 : size(initial_branch,2)
    upper_ = UB_fh(line_pair_data,initial_branch(:,i),epsilon_r,sample_reso,ids,sat_buff);
    branch = [branch,[initial_branch(:,i);upper_]];
end
% --- 3. Start the Acclerated BnB process ---
best_branch = [];
iter = 1;
while ~isempty(branch)
    % pop out the branch with highest upper bound
    [best_upper,max_idx] = max(branch(5,:));
    popped_branch = branch(:,max_idx);
    branch(:,max_idx)=[];
    [lower_bound,~] = LB_fh(line_pair_data,popped_branch(1:4),epsilon_r,ids,sat_buff,prox_thres);

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
    if popped_branch(3)-popped_branch(1)+10^(-8)<branch_reso 
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
            new_upper = UB_fh(line_pair_data,new_branch(:,i),epsilon_r,sample_reso,ids,sat_buff);
            if new_upper+eps>best_lower
               branch=[branch,[new_branch(:,i);new_upper]];
            end
        end
    end
end

% --- 4. Output ---
time=toc;
R_opt=[];
num_candidate=0;
for i = 1 : size(best_branch,2)
    this_branch = best_branch(:,i);
    this_u      = polar_2_xyz(0.5*(this_branch(1)+this_branch(3)) , 0.5*(this_branch(2)+this_branch(4)));
    [~,theta_opt] = LB_fh(line_pair_data,this_branch,epsilon_r,ids,sat_buff,prox_thres);
    theta_opt = cluster_stabber(theta_opt,prox_thres);
    for j = 1:length(theta_opt)
        append_R = rotvec2mat3d(this_u*theta_opt(j));
        R_opt=[R_opt;append_R'];
    end
    num_candidate = num_candidate + length(theta_opt);
end
end
