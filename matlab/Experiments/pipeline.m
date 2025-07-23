%%%%
% complete pipeline with Saturated Consensus Maximization
% --- Note!! --- 
% If you don't want to or can't use the compiled mex functions, 
% remeber to set variables 'mex_flag=0' in functions Sat_RotFGO and Sat_TransFGO

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT

clear
clc
scene_idx = 1;
pred_flag = 1;
%
room_sizes =  [ 8,    6, 4;
                 7,   7, 3;  
              10.5, 5, 3.5; 
              10.5, 6, 3.0];
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
dataset_name = dataset_names(scene_idx);
space_size =  room_sizes(scene_idx,:);
if pred_flag
    data_folder="csv_dataset/"+dataset_name+"_pred/";
    remapping = load(data_folder+"remapping.txt");
else
    data_folder="csv_dataset/"+dataset_name+"/";
    remapping=[];
end
lines3D=readmatrix(data_folder+"/3Dlines.csv"); 
%%% rot params
prox_thres_r = 1*pi/180; % for clustering proximate stabbers
branch_reso_r = pi/256; % terminate bnb when branch size < branch_reso
sample_reso_r = pi/256; % resolution for interval analysis
epsilon_r = 0.015;
q_rot = [0.7,0.6,0.6,0.9];
L_rot = q_rot(scene_idx)/(1-q_rot(scene_idx))/epsilon_r;
%%% trans params
branch_reso_t = 0.01; % terminate bnb when branch size <= branch_reso
prox_thres_t  = 0.01; %
epsilon_t = 0.03;
q_trans = 0.8;
L_trans = q_trans/(1-q_trans)/epsilon_t;
%%% statistics
column_names=["Image Id","# 2D lines","Outlier Ratio","IR Err Rot","IR Err Trans", "Rot Err","Trans Err", "time"];
columnTypes = ["int32"  ,"int32"     ,"double"       ,"double"    ,"double",       "double" ,"double"   ,"double"];
total_img=2000;
Record_est_SCM_entropy = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
parfor num =0:total_img
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_idx=num*10;     
    frame_id = sprintf("%06d",img_idx);
    if ~exist(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv",'file')     %%% read 2D line data of cur image
        continue
    end
    K_p=readmatrix(data_folder+"intrinsics/frame_"+frame_id+".csv");
    T_gt = readmatrix(data_folder+"poses/frame_"+frame_id+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1]; % intrinsic matrix
    % lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1) 
    lines2D = readmatrix(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv"); 
    lines2D(lines2D(:,4)==0,:)=[]; %delete lines without a semantic label
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    %
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/frame_"+frame_id+".csv")+1; % retrived sub-map
    retrived_closest_pose = readmatrix(data_folder+"retrived_closest_pose/frame_"+frame_id+".csv");
    [alpha,phi,theta] = rot2angle(retrived_closest_pose(1:3,1:3)');
    IR_err_rot = angular_distance(retrived_closest_pose(1:3,1:3),R_gt);
    IR_err_trans = norm(t_gt-retrived_closest_pose(1:3,4));
    %
    if abs(phi) < 3*pi/180    % ambiguous case
        west_east_flag = 2;
    else
        if phi<0
           west_east_flag = 1;
        else
           west_east_flag = 0;
        end
    end
    lines3D_sub = lines3D(retrived_3D_line_idx,:);
    [lines2D,lines3D_sub]=remap_semantic_id(lines2D,lines3D_sub,remapping);
    % ---------------------------------------------------------------------
    % --- 2. semantic matching and outlier ratio ---
    [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D_sub);  % match with clustered 3D lines
    % skip image with too few 2D lines
    if length(unique(ids)) < 5
       fprintf("image "+num2str(img_idx)+" has less than 5 lines, skip.\n");
       continue
    end
    total_match_num = size(n_2D,1);
    with_match_ids = find(lines2D(:,9)>0);
    if pred_flag
       predicted_semantics = lines2D(with_match_ids,4);
       true_semantics = lines3D(lines2D(with_match_ids,9),7);
       outlier_ratio = 1-nnz(predicted_semantics==true_semantics)/total_match_num;
    else
       outlier_ratio = 1-length(with_match_ids)/total_match_num;
    end

    %-------------------------------------------------------------
    %---- 3. complete pipeline starts here -----
    fprintf(num2str(img_idx)+"\n")
    time_all = 0; 
    %%%%%%%%%%%% rotation estimation %%%%%%%%%%%% 
    % saturation function design
    num_2D_lines = size(lines2D,1);
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(ids==i);
    end
    rot_kernel_buff_SCM_entropy = zeros(num_2D_lines,max(match_count));
    for i = 1:num_2D_lines
        if match_count(i)==0
            continue
        end
        rot_kernel_buff_SCM_entropy(i,1) =  log(1+L_rot/match_count(i));
        for j = 2:match_count(i)
            rot_kernel_buff_SCM_entropy(i,j) = log(1+L_rot*j/match_count(i))-log(1+L_rot*(j-1)/match_count(i));
        end
    end
    gt_inliers_idx = find(abs(dot(R_gt'*v_3D',n_2D'))<=epsilon_r);
    gt_inliers_id = ids(gt_inliers_idx);
    [R_opt_top,best_score,num_candidate_rot,time,~,~] = ...
        Sat_RotFGO(n_2D,v_3D,ids,rot_kernel_buff_SCM_entropy,...
        branch_reso_r,epsilon_r,sample_reso_r,prox_thres_r,west_east_flag);
    time_all = time_all+time;
    best_t_score = -1;
    best_R = eye(3);
    best_t = zeros(3,1);
    for n = 1:num_candidate_rot
        R_opt = R_opt_top(n*3-2:n*3,:)';
        [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
            under_specific_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r);
        % observability check
        ambiguiFlag = checkTransAmbiguity(img_idx,lines2D(unique(id_inliers_under_rot),:),R_gt);
        if ambiguiFlag
            continue
        end
        %%% saturation function design
        match_count_pruned = zeros(num_2D_lines,1);
        for i = 1:num_2D_lines
            match_count_pruned(i) = sum(id_inliers_under_rot==i);
        end
        LL = sum(log(match_count_pruned(match_count_pruned>0)));
        trans_kernel_buff_SCM_entropy = zeros(num_2D_lines,max(match_count_pruned));
        for i = 1:num_2D_lines
            if match_count_pruned(i)==0
                continue
            end
            trans_kernel_buff_SCM_entropy(i,1)=log(1+L_trans/match_count(i));
            for j = 2:match_count_pruned(i)
                trans_kernel_buff_SCM_entropy(i,j)=log(1+L_trans*j/match_count(i))-log(1+L_trans*(j-1)/match_count(i));
            end
        end
        %%% Sat-CM: entropy + geo 
        [t_best_candidates,~,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,trans_kernel_buff_SCM_entropy,space_size,branch_reso_t,epsilon_t,prox_thres_t);
        time_all = time_all+time;
        % prune candidates according to geometric constraints
        [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,trans_kernel_buff_SCM_entropy);
        t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
        if best_score > best_t_score
            best_t_score = best_score;
            best_R = R_opt;
            best_t = t_fine_tuned;
        end
    end
    rot_err = angular_distance(best_R,R_gt);
    t_err   = norm(best_t-t_gt);
    column_names=["Image Id","# 2D lines","Outlier Ratio","IR Err Rot","IR Err Trans", "Rot Err","Trans Err", "time"];
    Record_est_SCM_entropy(num+1,:)={img_idx,num_2D_lines,outlier_ratio,IR_err_rot,IR_err_trans,rot_err,t_err,time_all};
end
Record_est_SCM_entropy(Record_est_SCM_entropy.("Outlier Ratio")==0,:)=[];
%% 
if pred_flag
    output_filename= "./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_full_record.mat";
else
    output_filename= "./matlab/Experiments/records/gt_semantics/"+dataset_name+"_full_record.mat";
end
save(output_filename);
num_valid = height(Record_est_SCM_entropy);
fprintf("============ time statistics ============\n")
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_est_SCM_entropy.time,[0.25,0.5,0.75]))
fprintf("============ rot err quantile ============\n")
fprintf("IR: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("IR Err Rot"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("Rot Err"),[0.25,0.5,0.75]))
fprintf("============ trans err quantile ============\n")
fprintf("IR: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("IR Err Trans"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(Record_est_SCM_entropy.("Trans Err"),[0.25,0.5,0.75]))
fprintf("============ Recall at 3/5/10 deg ============\n")
fprintf("IR: %f,%f,%f\n",sum(Record_est_SCM_entropy.("IR Err Rot")<[3,5,10])/num_valid*100)
fprintf("SCM_FGO_entropy: %f,%f,%f\n",sum(Record_est_SCM_entropy.("Rot Err")<[3,5,10])/num_valid*100)
fprintf("============ Recall at 5cm/10cm/20cm ============\n")
fprintf("IR: %f,%f,%f\n",sum(Record_est_SCM_entropy.("IR Err Trans")<[0.05,0.1,0.2])/num_valid*100)
fprintf("SCM_FGO_entropy: %f,%f,%f\n",sum(Record_est_SCM_entropy.("Trans Err")<[0.05,0.1,0.2])/num_valid*100)
% %
% ---------------------------------------------------------------------
% --- sub-functions ---
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


function [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
    under_specific_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r)

    inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
    id_inliers_under_rot = ids(inlier_under_rot);
    n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
    endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
    %%% fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
    pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
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
