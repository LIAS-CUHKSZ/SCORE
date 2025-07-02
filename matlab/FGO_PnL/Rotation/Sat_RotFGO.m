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

function [R_opt,best_lower,num_candidate,time,upper_record,lower_record] = Sat_RotFGO(vector_n,vector_v,ids,kernel_buff,branch_reso,epsilon_r,sample_reso,prox_thres)

mex_flag = 1; %set true to use mex code for acceleration.
verbose_flag = 0; %set true for detailed bnb process info.
% ---------------------------------------------------------------------
% --- 1. preprocess ---
line_pair_data = data_process(vector_n,vector_v); % pre-process data

% ---------------------------------------------------------------------
% --- 2. Initialize the Acclerated BnB process ---
tic
%  calculate bounds for two semispheres.
branch=[];
B_east=[0;0;pi;pi]; B_west=[0;pi;pi;2*pi];
if mex_flag
    [upper_east,lower_east,theta_east]=Sat_Bounds_FGO_mex(line_pair_data,B_east,epsilon_r,sample_reso,ids,kernel_buff,prox_thres);    
    [upper_west,lower_west,theta_west]=Sat_Bounds_FGO_mex(line_pair_data,B_west,epsilon_r,sample_reso,ids,kernel_buff,prox_thres);
else
    [upper_east,lower_east,theta_east]=Sat_Bounds_FGO(line_pair_data,B_east,epsilon_r,sample_reso,ids,kernel_buff,prox_thres);    
    [upper_west,lower_west,theta_west]=Sat_Bounds_FGO(line_pair_data,B_west,epsilon_r,sample_reso,ids,kernel_buff,prox_thres);
end
branch=[branch,[B_east;upper_east;lower_east]];
branch=[branch,[B_west;upper_west;lower_west]];
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
upper_record=best_upper; lower_record=best_lower; % record bounds history
new_upper=zeros(1,4); new_lower=zeros(1,4); 
new_theta_lower=cell(1,4); % it is possible that multiple optimal stabbers returned by Sat-IS
iter=1;

% ---------------------------------------------------------------------
% --- 3. Start the Acclerated BnB process ---
while true
    new_branch=subBranch(next_branch);
    for i=1:4
        if mex_flag
            [new_upper(i),new_lower(i),new_theta_lower{i}]=Sat_Bounds_FGO_mex(line_pair_data,new_branch(:,i),epsilon_r,sample_reso,ids,kernel_buff,prox_thres);
        else
            [new_upper(i),new_lower(i),new_theta_lower{i}]=Sat_Bounds_FGO(line_pair_data,new_branch(:,i),epsilon_r,sample_reso,ids,kernel_buff,prox_thres);
        end
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
