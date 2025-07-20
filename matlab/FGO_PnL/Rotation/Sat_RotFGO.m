%%%%
% Implementation of the FGO rotation estimator under saturated consensus maximization

%%% Inputs:
% vector_n:         N x 3, the normal vector paramater of 2D image line for each association.
% vector_v:         N x 3, the direction vector of 3D map line for each association.
% ids:              N x 1, the belonging 2D line id for each association.
% kernel_buffer:    stored weights given by the saturation function.
% branch_reso:      scalar, stop bnb when cube length < resolution.
% epsilon_r:        scalar, error tolerance.
% sample_reso:      scalar, control resolution for interval analysis.
% prox_thres:       double, the candidates are not proximate to each other.
% west_or_east:     =0 only branch over the west sphere
%                   =1 only branch over the east sphere
%                   =2 branch over the whole sphere

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [R_opt,best_lower,num_candidate,time,upper_record,lower_record] = Sat_RotFGO(vector_n,vector_v,ids,kernel_buff,branch_reso,epsilon_r,sample_reso,prox_thres, west_or_east)

mex_flag = 1; %set true to use mex code for acceleration.
verbose_flag = 0; %set true for detailed bnb process info.
% choose the function handler according to mex_flag
if mex_flag
    Bound_fh = @Sat_Bounds_FGO_mex;
else
    Bound_fh = @Sat_Bounds_FGO;
end

% ---------------------------------------------------------------------
% --- 1. preprocess ---
line_pair_data = data_process(vector_n,vector_v); % pre-process data

% ---------------------------------------------------------------------
% --- 2. Initialize the Acclerated BnB process ---
tic
%  initialization
best_lower = -1; best_upper = -1;
branch=[];
upper_record=[]; lower_record=[]; % record bounds history
if west_or_east == 2 % branch over the whole sphere
    % bound the east semi-sphere first
    east_branch = [0;0;pi;pi];
    [upper,lower,theta]=Bound_fh(line_pair_data,east_branch,epsilon_r,sample_reso,ids,kernel_buff,prox_thres);

    % optimal rotation axis (corresponding to the branch achieving the best lower bound)
    u_best=polar_2_xyz(0.5*(east_branch(1)+east_branch(3)) , 0.5*(east_branch(2)+east_branch(4)) );

    % cluster the optimal stabbers returned by saturated interval stabbing
    theta_best = cluster_stabber(theta,prox_thres);
    u_best=repmat(u_best,1,length(theta_best));

    % update best lower/upper bound
    best_lower = lower; best_upper = upper;
    branch = [east_branch;upper;lower];

    % set the west semi-sphere as the next branch to be branched and bounded
    next_branch = [0;pi;pi;2*pi];
else
    if west_or_east % only branch over the west sphere
        % set the west semi-sphere as the next branch to be branched and bounded
        next_branch = [0;pi;pi;2*pi];
    else % only branch over the east sphere
        % set the east semi-sphere as the next branch to be branched and bounded
        next_branch = [0;0;pi;pi];
    end
end

% ---------------------------------------------------------------------
% --- 3. Start the Acclerated BnB process ---
new_upper=zeros(1,4); new_lower=zeros(1,4); new_theta_lower=cell(1,4);
while true
    % divide the branch into four
    new_branch=subBranch(next_branch); 
    % obtain lower/upper bound and optimal stabbers for each sub branch
    for i=1:4
        [new_upper(i),new_lower(i),new_theta_lower{i}]=Bound_fh(line_pair_data,new_branch(:,i),epsilon_r,sample_reso,ids,kernel_buff,prox_thres);
        if(verbose_flag)
            fprintf('Iteration: %d, Branch: [%f, %f, %f, %f], Upper: %f, Lower: %f\n', iter, new_branch(:,i), new_upper(i), new_lower(i));
        end
    end
    branch=[branch,[new_branch;new_upper;new_lower]];

    % update the best lower bound and the optimal rotation candidates
    sub_max_lower = max(new_lower);
    idx_sub_best = find(new_lower==sub_max_lower);
    for i=1:length(idx_sub_best)
        j = idx_sub_best(i);
        if  sub_max_lower>best_lower
            best_lower=sub_max_lower;
            % cluster the optimal stabbers returned by saturated interval stabbing
            theta_best = cluster_stabber(new_theta_lower{j},prox_thres);

            % update best rotation axis
            u_best=polar_2_xyz(0.5*(new_branch(1,j)+new_branch(3,j)) , 0.5*(new_branch(2,j)+new_branch(4,j)) );
            u_best=repmat(u_best,1,length(theta_best));

        elseif sub_max_lower==best_lower
            % cluster the optimal stabbers returned by saturated interval stabbing
            new_best_theta = cluster_stabber(new_theta_lower{j},prox_thres);

            % append the buffer storing optimal rotation axis(u_best) and angle(theta_best) 
            u_new=polar_2_xyz(0.5*(new_branch(1,j)+new_branch(3,j)),0.5*(new_branch(2,j)+new_branch(4,j)));
            u_best = [u_best,repmat(u_new,[1,length(new_best_theta)])];
            theta_best = [theta_best,new_best_theta];
            
        else
            contiune;
        end
    end
    branch(:,branch(5,:)<best_lower)=[]; % prune branches

    % update the best upper bound and determine the next branch
    best_upper = max(branch(5,:));
    % choose the branch with best upper bound and largest size
    idx_upper = find(branch(5,:)==best_upper); % find branches achieving the best upper bound
    branch_size=branch(3,idx_upper)-branch(1,idx_upper); 
    [~,temp_idx]=max(branch_size);
    idx_upper=idx_upper(temp_idx); % choose the one with largest size
    next_branch=branch(1:4,idx_upper);
    branch(:,idx_upper)=[];
   
    % stop condition
    if (  (next_branch(3,1) - next_branch(1,1)) < branch_reso )
        break;
    end

    % record the history of best lower/upper bounds
    lower_record=[lower_record;best_lower];
    upper_record=[upper_record;best_upper];
end

% ---------------------------------------------------------------------
% --- 4. Output ---
time=toc;
num_candidate=size(u_best,2);
R_opt=[];
for i=1:num_candidate
    R_opt = [R_opt;rotvec2mat3d(u_best(:,i)*theta_best(i))];
end
end
