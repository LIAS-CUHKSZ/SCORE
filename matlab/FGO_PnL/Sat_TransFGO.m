%%%%
% Implementation of the FGO translation estimator under saturated consensus maximization

%%% Inputs:
% pert_rot_n_2D:N  x 3, the rotated and perturbed normal vector paramater for each 2D image line.
% endpoints_3D: 2N x 3, the endpoints of 3D lines
% space_size:   3  x 1, the space bounding box
% ids:          N x 1, the belonging 2D line id for each matched pair.
% kernel_buffer: store weights given by the selected saturation function.
% branch_reso: scalar, stop bnb when cube length < resolution.
% epsilon_t: scalar, for Sat-CM formulation.
% sample_reso: scalar, control resolution for interval analysis.
% prox_thres:  double, the candidates are not proximate to each other.
% verbose_flag: bool, set true for detailed bnb process info.
% mex_flag: bool, set true to use mex code for acceleration.



%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [t_best,best_lower,num_candidate,time,upper_record,lower_record] = Sat_TransFGO(pert_rot_n_2D,endpoints_3D,ids,kernel_buff,space_size,branch_reso,epsilon_t,sample_reso,verbose_flag,mex_flag)
%%%%%%%%%%%%%%%%%%%%% Acclerated BnB %%%%%%%%%%%%%%%%%%%%%
tic
%%% Initialize the BnB process
% reduce (x) and divide (y,z) into cubes of 1 meters
ceil_y = ceil(space_size(2)); ceil_z = ceil(space_size(3));
branch = zeros(6,ceil_y*ceil_z); 
best_lower = -1; t_best=zeros(3,1);
for i=1:ceil_y
    for j=1:ceil_z
        idx = (i-1)*ceil_z+j;
        br_ = [i-1;j-1;i;j];
        [u_,l_,t_sample] = Sat_Bounds_trans(pert_rot_n_2D,endpoints_3D(1:2:end,:),ids,epsilon_t,br_,sample_reso,space_size,kernel_buff);
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
upper_record=[]; lower_record=[];
%%% start BnB
iter=1;
while true
    % record bounds history
    best_upper  = max(branch(5,:));
    upper_record=[upper_record;best_upper];
    lower_record=[lower_record;best_lower];
    % select the next exploring branch according to the upper bounds
    idx_upper   = find(branch(5,:)==best_upper);
    branch_size = branch(3,idx_upper)-branch(1,idx_upper);
    [~,temp_idx]= max(branch_size); idx_upper=idx_upper(temp_idx);
    next_branch=branch(1:4,idx_upper); % choose the one with largest upper bound and largest width
    branch(:,idx_upper)=[];
    branch(:,branch(5,:)<best_lower)=[];
    if (  (next_branch(3,1) - next_branch(1,1)) < branch_reso )
        break;
    end
    iter=iter+1;
    new_branch=subBranch(next_branch);
    new_upper = zeros(1,4); new_lower = zeros(1,4); new_t_sample = cell(4,1);
    for i=1:4
        [new_upper(i),new_lower(i),new_t_sample{i}]=Sat_Bounds_trans(pert_rot_n_2D,endpoints_3D(1:2:end,:),ids,epsilon_t,new_branch(:,i),sample_reso,space_size,kernel_buff);
        if verbose_flag
            fprintf('Iteration: %d, Branch: [%f, %f, %f, %f], Upper: %f, Lower: %f\n', iter, new_branch(:,i), new_upper(i), new_lower(i));
        end
    end
    branch=[branch,[new_branch;new_upper;new_lower]];
    for i=1:4
        if  new_lower(i)>best_lower
            best_lower=new_lower(i);
            t_best = new_t_sample{i};
            continue;
        end
        if new_lower(i)==best_lower
            t_best = [t_best, new_t_sample{i}];
        end
    end
end
%%% output
num_candidate=size(t_best,2);
time=toc;
end
