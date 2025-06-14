%%%%
% Orientation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% GT rotation vs FGOPnL estimates

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%
clear
clc
room_sizes = [10.3, 6, 2.6;  
              5.8,  4, 3.9; 
              10.4, 5, 3.3; 
              7,    7, 2.9];
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(3);
space_size =  room_sizes(3,:);
data_folder="csv_dataset/"+dataset_idx+"/";
load(data_folder+"lines3D.mat");
%%% orientation params
% kernel_SCM = @(x) x^(-4);
kernel_SCM_1 = @(x) 1-(x>1);
kernel_SCM_2 = @(x) x^(-2); % @(x) 2^(-x+1);  
trunc_num=length(lines3D);  
kernel_buff_SCM_1=zeros(trunc_num,1); kernel_buff_SCM_2=zeros(trunc_num,1);
for i=1:trunc_num
    kernel_buff_SCM_1(i)=kernel_SCM_1(i);
    kernel_buff_SCM_2(i)=kernel_SCM_2(i);
end
total_img=1000;
%%% translation params
verbose_flag=0; % verbose mode for BnB 
mex_flag=1; % use matlab mex code for acceleration
branch_reso = 0.05; % terminate bnb when branch size <= branch_reso
sample_reso = 0.03; % resolution for interval analysis
%%
column_names=["Image Id","# 2D lines","epsilon_t","Outlier Ratio","Rot Err","Max Trans Err","Min Trans Err","# Trans Candidates","BnB Score","GT Score","time"];
columnTypes = ["int32","int32","double","double","double","double","double","int32","double","double","double"];
Record_gt_SCM =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO = load("matlab\Experiments\records\"+dataset_idx+"_rotation_record.mat").("Record_SCM_FGO");
epsilon_rs = Record_SCM_FGO.epsilon_r;
%%% Experiments
valid_idx = Record_SCM_FGO{:,1}; % These images have passed the rotation ambiguity test.
parfor num =1:length(valid_idx)
    img_idx=valid_idx(num);     
    %%% read 2D line data of cur image
    frame_id = sprintf("%06d",img_idx);
    K_p=readmatrix(data_folder+"intrinsics\frame_"+frame_id+".csv");
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(data_folder+"poses\frame_"+frame_id+".csv"); R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    % lines2D(Nx9): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1) 
    lines2D = readmatrix(data_folder+"lines2D\frame_"+frame_id+"_2Dlines.csv"); 
    lines2D = lines2D(lines2D(:,4)~=0,:);   % delete 2D line without a semantic label
    %%%%% note: annotating the below line leads to a more challenging problem setting 
    lines2D = lines2D(lines2D(:,9)>=0,:); % delete 2D line without a real matching
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic;  lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    %%% check translation ambiguity
    ambiguiFlag = checkTransAmbiguity(img_idx,lines2D,R_gt);
    if ambiguiFlag
        continue
    end
    %%% match 2D and 3D lines using semnatic label
    M = size(lines2D,1);
    [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D);  % match with unclustered 3D lines 
    epsilon_r = epsilon_rs(num);
    epsilon_t = max(lines2D(:,11))+0.001;
    %%%%%%%%%%%%%%%%%%% Estimate Translation with GT Orient
    R_opt = R_gt;
    inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
    id_inliers_under_rot = ids(inlier_under_rot);
    if length(unique(id_inliers_under_rot))<4
        continue
    end
    n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
    endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
    %%% fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
    pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
    residuals = sum(pert_rot_n_2D_inlier.*(endpoints_3D_inlier(1:2:end,:)-t_gt'),2);
    inliers_t_gt = find(abs(residuals )<=epsilon_t);
    inliers_t_gt = prune_inliers(R_opt,intrinsic,inliers_t_gt,endpoints_3D_inlier,t_gt);
    score_t_gt = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_SCM_2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [t_best_candidates,best_score_1,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_1,space_size,branch_reso,epsilon_t,sample_reso,verbose_flag,mex_flag);
    % prune candidates according to physical constraints
    [best_score_2,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM_2);
    fprintf("Image"+num2str(img_idx)+"\n")
    if num_candidate ~= size(t_best_candidates,2)
       fprintf("%d candidates pruned to %d \n",num_candidate,size(t_best_candidates,2));
       num_candidate = size(t_best_candidates,2);
    end
    [min_err,max_err,t_min]=min_max_trans_error(num_candidate,t_best_candidates,t_gt);
    Record_gt_SCM(num+1,:)={img_idx,M,epsilon_t,1-M/size(n_2D_inlier,1),0,max_err,min_err,num_candidate,best_score_2,score_t_gt,time};
end
Record_gt_SCM(Record_gt_SCM.("BnB Score")==0,:)=[];
output_filename= "./matlab/Experiments/records/"+dataset_idx+"_translation_record.mat";
save(output_filename);
% Record_gt_SCM(Record_gt_SCM.("# 2D lines")<Record_gt_SCM.("GT Score"),:)=[];

%%%%%%%%%%%%% sub-functions %%%%%%%%%%%%%
function flag = checkTransAmbiguity(img_idx,lines2D,R_gt)
    M = size(lines2D,1);
    A_gt = zeros(M,3);
    flag = false;
    for i=1:M
        n = lines2D(i,1:3); 
        A_gt(i,:)=(R_gt*n')';
    end
    if rank(A_gt'*A_gt)<3
        flag = true;
        fprintf("image"+num2str(img_idx)+" is ambigious in translation, skip.\n");
    end
end

function [best_score,t_real_candidate] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D,endpoints_3D,ids,epsilon_t,t_candidates,kernel_buff)
    p_3D = endpoints_3D(1:2:end,:);
    best_score = -1; t_real_candidate=[];
    for k=1:size(t_candidates,2)
        t_test = t_candidates(:,k);
        residuals = sum(pert_rot_n_2D.*(p_3D-t_test'),2);
        inliers = find(abs(residuals )<=epsilon_t);
        %%% the above inliers satisfy the geometric constraints,
        %%% we urther filter lines behind the camera and outside image
        real_inliers = prune_inliers(R_opt,intrinsic,inliers,endpoints_3D,t_test);
        score=calculate_score(ids(real_inliers),kernel_buff);
        if score > best_score
            best_score = score; t_real_candidate = t_test;
        elseif score == best_score
            t_real_candidate = [t_real_candidate,t_test];
        else
        end
    end
end
