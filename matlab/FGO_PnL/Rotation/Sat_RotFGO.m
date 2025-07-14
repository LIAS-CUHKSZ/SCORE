%%%%
% Implementation of the FGO rotation estimator under saturated consensus maximization

%%% Inputs:
% vector_n:         N x 3, the normal vector paramater for each 2D image line.
% vector_v:         N x 3, the direction vector for each matched 3D map line.
% ids:              N x 1, the belonging 2D line id for each matched pair.
% kernel_buffer:    store weights given by the saturation function.
% branch_reso:      scalar, stop bnb when cube length < resolution.
% epsilon_r:        scalar, error tolerance.
% sample_reso:      scalar, control resolution for interval analysis.
% prox_thres:       double, the candidates are not proximate to each other.
%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT
%%%%

function [R_opt,best_lower,num_candidate,time,upper_record,lower_record] = Sat_RotFGO(vector_n,vector_v,ids,kernel_buff,branch_reso,epsilon_r,sample_reso,prox_thres, west_or_east)

mex_flag = 1; %set true to use mex code for acceleration.
verbose_flag = 0; %set true for detailed bnb process info.
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
if west_or_east == 2
   east_branch = [0;0;pi;pi];
   [upper,lower,theta]=Bound_fh(line_pair_data,east_branch,epsilon_r,sample_reso,ids,kernel_buff,prox_thres);
   u_best=polar_2_xyz(0.5*(east_branch(1)+east_branch(3)) , 0.5*(east_branch(2)+east_branch(4)) );
   theta_best = cluster_stabber(theta,prox_thres);  
   u_best=repmat(u_best,1,length(theta_best));
   best_lower = lower; best_upper = upper;
   branch = [east_branch;upper;lower];
   next_branch = [0;pi;pi;2*pi];
else
    if west_or_east
        next_branch = [0;pi;pi;2*pi];
    else
       next_branch = [0;0;pi;pi];
    end
end
new_upper=zeros(1,4); new_lower=zeros(1,4); 
new_theta_lower=cell(1,4); % it is possible that multiple optimal stabbers returned by Sat-IS
iter=1;
% ---------------------------------------------------------------------
% --- 3. Start the Acclerated BnB process ---
while true
    new_branch=subBranch(next_branch);
    for i=1:4
        [new_upper(i),new_lower(i),new_theta_lower{i}]=Bound_fh(line_pair_data,new_branch(:,i),epsilon_r,sample_reso,ids,kernel_buff,prox_thres);
        if(verbose_flag)
            fprintf('Iteration: %d, Branch: [%f, %f, %f, %f], Upper: %f, Lower: %f\n', iter, new_branch(:,i), new_upper(i), new_lower(i));
        end
    end
    branch=[branch,[new_branch;new_upper;new_lower]];
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
            theta_best = [theta_best,new_best_theta];
            u_best = [u_best,repmat(u_new,[1,length(new_best_theta)])];
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

% ---------------------------------------------------------------------
% --- 4. Output ---
time=toc;
num_candidate=size(u_best,2);
R_opt=[];
for i=1:num_candidate
    R_opt = [R_opt;rotvec2mat3d(u_best(:,i)*theta_best(i))];
end
end
