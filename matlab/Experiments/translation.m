%%%%
% Translation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% --- Note!! --- 
% If you don't want to or can't use the compiled mex functions, 
% remeber to set variables 'mex_flag=0' in function Sat_TransFGO

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT



clear
clc
scene_idx = 2;
pred_flag = 1; % set 1 if use predicted semantic label
%
room_sizes =  [8,    6, 4;
               7,   7, 3;  
              10.5, 5, 3.5; 
              10.5, 6, 3.0];
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
dataset_name = dataset_names(scene_idx);
space_size =  room_sizes(scene_idx,:);
if pred_flag
    data_folder="csv_dataset/"+dataset_name+"_pred/";
else
    data_folder="csv_dataset/"+dataset_name+"/";
end
lines3D=readmatrix(data_folder+"/3Dlines.csv"); 
%%% params
branch_reso = 0.01; % terminate bnb when branch size <= branch_reso
prox_thres  = 0.01; %  
%%% statistics
column_names= ["Image Id","# 2D lines","epsilon_t","Outlier Ratio","IR Err Trans","Rot Err","Trans Err","BnB Score","GT Score","time"];
columnTypes = ["int32",  "int32",     "double",   "double",       "double",        "double","double","double","double","double"];
total_img=2000;
Record_gt_CM = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_gt_SCM_power = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_gt_SCM_exp = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_gt_SCM_entropy = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
%%% rotation data
epsilon_r = 0.015;
epsilon_t = 0.03;
%% 
parfor num = 0:2000
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_idx=num*10;        
    frame_id = sprintf("%06d",img_idx);
    if ~exist(data_folder+"lines2D/frame_"+frame_id+"_2Dlines.csv",'file')     %%% read 2D line data of cur image
        continue
    end
    K_p=readmatrix(data_folder+"intrinsics\frame_"+frame_id+".csv");
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(data_folder+"poses\frame_"+frame_id+".csv"); R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/frame_"+frame_id+".csv")+1;
    retrived_closest_pose = readmatrix(data_folder+"retrived_closest_pose/frame_"+frame_id+".csv");
    retrived_err = norm(retrived_closest_pose(1:3,4)-t_gt);
    % lines2D(Nx9): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1) 
    lines2D = readmatrix(data_folder+"lines2D\frame_"+frame_id+"_2Dlines.csv"); 
    lines2D(lines2D(:,4)==0,:)=[]; % delete lines without semantic label
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic;  lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    num_2D_lines = size(lines2D,1);

    % --- 2. semantic matching and observability check
    lines3D_sub = lines3D(retrived_3D_line_idx,:);
    [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D_sub);  
    ambiguiFlag = checkTransAmbiguity(img_idx,lines2D(unique(ids),:),R_gt);
    if ambiguiFlag
        continue
    end
    fprintf("Image"+num2str(img_idx)+"\n")

    % --- 3. find rotation inlier and calculate outlier ratio ---
    R_opt = R_gt; % we use ground truth rotation in this experiment
    [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
        under_specific_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r);
    total_match_num = size(pert_rot_n_2D_inlier,1);
    unique_id_set = unique(id_inliers_under_rot);
    with_match_id_set = unique_id_set(lines2D(unique_id_set,9)>0);
    if pred_flag
        predicted_semantics = lines2D(with_match_id_set,4);
        true_semantics   = lines3D(lines2D(with_match_id_set,9),7);
        outlier_ratio = 1 - nnz(predicted_semantics==true_semantics)/total_match_num;
    else
        outlier_ratio = 1 - length(with_match_id_set)/total_match_num;
    end

    % --- 4. saturation function design  ---
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(id_inliers_under_rot==i);
    end
    L = sum(log(match_count(match_count>0)));
    kernel_buff_SCM_entropy = zeros(num_2D_lines,max(match_count));
    kernel_buff_SCM_power = zeros(num_2D_lines,max(match_count));
    kernel_buff_SCM_exp =  zeros(num_2D_lines,max(match_count));
    kernel_buff_CM  = ones(num_2D_lines,max(match_count));
    for i = 1:num_2D_lines
        if match_count(i)==0
            continue
        end
        kernel_buff_SCM_power(i,1)=1;
        kernel_buff_SCM_exp(i,1)=1;
        kernel_buff_SCM_entropy(i,1)=1-log(match_count(i))/L;
        for j = 2:match_count(i)
            kernel_buff_SCM_entropy(i,j)=(log(j)-log(j-1))/L;
            kernel_buff_SCM_power(i,j) = j^(-8);
            kernel_buff_SCM_exp(i,j) = 2^(-j);
        end
    end
    kernel_buff_SCM_power(:,2:end)=kernel_buff_SCM_power(:,2:end)/sum(sum(kernel_buff_SCM_power(:,2:end)));
    kernel_buff_SCM_exp(:,2:end)=kernel_buff_SCM_exp(:,2:end)/sum(sum(kernel_buff_SCM_exp(:,2:end)));

    % --------------------------------------
    % --- 5. translation estimation starts here ---
    residuals = sum(pert_rot_n_2D_inlier.*(endpoints_3D_inlier(1:2:end,:)-t_gt'),2);
    inliers_t_gt = find(abs(residuals )<=epsilon_t);

    %%% CM    
    score_t_gt_CM = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_CM);
    [t_best_candidates,best_score,~,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,space_size,branch_reso,epsilon_t,prox_thres);
    % prune candidates according to geometric constraints
    [~,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_CM);
    t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
    t_err = norm(t_fine_tuned-t_gt);
    Record_gt_CM(num+1,:)={img_idx,M,epsilon_t,outlier_ratio,retrived_err,0,t_err,best_score,score_t_gt_CM,time};

    %%% Sat-CM: power function 
    score_t_gt_SCM_power = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_SCM_power);
    [t_best_candidates,best_score,~,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_power,space_size,branch_reso,epsilon_t,prox_thres);
    % prune candidates according to geometric constraints
    [~,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM_power);
    t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
    t_err = norm(t_fine_tuned-t_gt);
    Record_gt_SCM_power(num+1,:)={img_idx,M,epsilon_t,outlier_ratio,retrived_err,0,t_err,best_score,score_t_gt_SCM_power,time};
    
    %%% Sat-CM: exponential function
    score_t_gt_SCM_exp = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_SCM_exp);
    [t_best_candidates,best_score,~,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_exp,space_size,branch_reso,epsilon_t,prox_thres);
    % prune candidates according to geometric constraints
    [~,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM_exp);
    t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
    t_err = norm(t_fine_tuned-t_gt);
    Record_gt_SCM_exp(num+1,:)={img_idx,M,epsilon_t,outlier_ratio,retrived_err,0,t_err,best_score,score_t_gt_SCM_exp,time};
    
    %%% Sat-CM: entropy function
    score_t_gt_SCM_entropy = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_SCM_entropy);
    [t_best_candidates,best_score,~,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_entropy,space_size,branch_reso,epsilon_t,prox_thres);
    % prune candidates according to geometric constraints
    [~,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM_entropy);
    t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
    t_err = norm(t_fine_tuned-t_gt);
    Record_gt_SCM_entropy(num+1,:)={img_idx,M,epsilon_t,outlier_ratio,retrived_err,0,t_err,best_score,score_t_gt_SCM_entropy,time};
end
Record_gt_CM(Record_gt_CM.("BnB Score")==0,:)=[];
Record_gt_SCM_power(Record_gt_SCM_power.("BnB Score")==0,:)=[];
Record_gt_SCM_exp(Record_gt_SCM_exp.("BnB Score")==0,:)=[];
Record_gt_SCM_entropy(Record_gt_SCM_entropy.("BnB Score")==0,:)=[];
%%
if pred_flag
    output_filename= "./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_translation_record.mat";
else
    output_filename= "./matlab/Experiments/records/gt_semantics/"+dataset_name+"_translation_record.mat";
end
save(output_filename);
num_valid_images = height(Record_gt_CM);
fprintf("============ statistics ============\n")
fprintf("num of valid images: %d\n",num_valid_images);
fprintf("num of re-localized images ( < 10 cm):\n")
fprintf("CM_gt_FGO: %d \n",length(find(Record_gt_CM.("Trans Err")<0.1)))
fprintf("SCM_gt_FGO_power: %d \n",length(find(Record_gt_SCM_power.("Trans Err")<0.1)))
fprintf("SCM_gt_FGO_exp: %d \n",length(find(Record_gt_SCM_exp.("Trans Err")<0.1)))
fprintf("SCM_gt_FGO_entropy: %d \n",length(find(Record_gt_SCM_entropy.("Trans Err")<0.1)))
%%
fprintf("============ time statistics (trans only) ============\n")
fprintf("CM_gt_FGO: %f,%f,%f\n",quantile(Record_gt_CM.("time"),[0.25,0.5,0.75]))
fprintf("SCM_gt_FGO_power: %f,%f,%f\n",quantile(Record_gt_SCM_power.("time"),[0.25,0.5,0.75]))
fprintf("SCM_gt_FGO_exp: %f,%f,%f\n",quantile(Record_gt_SCM_exp.("time"),[0.25,0.5,0.75]))
fprintf("SCM_gt_FGO_entropy: %f,%f,%f\n",quantile(Record_gt_SCM_entropy.("time"),[0.25,0.5,0.75]))

fprintf("============ Trans Err statistics ============\n")
fprintf("IR: %f,%f,%f\n",quantile(Record_gt_CM.("IR Err Trans"),[0.25,0.5,0.75]))
fprintf("CM_gt_FGO: %f,%f,%f\n",quantile(Record_gt_CM.("Trans Err"),[0.25,0.5,0.75]))
fprintf("SCM_gt_FGO_power: %f,%f,%f\n",quantile(Record_gt_SCM_power.("Trans Err"),[0.25,0.5,0.75]))
fprintf("SCM_gt_FGO_exp: %f,%f,%f\n",quantile(Record_gt_SCM_exp.("Trans Err"),[0.25,0.5,0.75]))
fprintf("SCM_gt_FGO_entropy: %f,%f,%f\n",quantile(Record_gt_SCM_entropy.("Trans Err"),[0.25,0.5,0.75]))

fprintf("============ Recall at 5cm/10cm/20cm ============\n")
fprintf("Image Retriveal: %f,%f,%f\n",sum(Record_gt_CM.("IR Err Trans")<[0.05,0.1,0.2])/num_valid_images*100)
fprintf("CM_gt_FGO: %f,%f,%f\n",sum(Record_gt_CM.("Trans Err")<[0.05,0.1,0.2])/num_valid_images*100)
fprintf("SCM_gt_FGO_power: %f,%f,%f\n",sum(Record_gt_SCM_power.("Trans Err")<[0.05,0.1,0.2])/num_valid_images*100)
fprintf("SCM_gt_FGO_exp: %f,%f,%f\n",sum(Record_gt_SCM_exp.("Trans Err")<[0.05,0.1,0.2])/num_valid_images*100)
fprintf("SCM_gt_FGO_entropy: %f,%f,%f\n",sum(Record_gt_SCM_entropy.("Trans Err")<[0.05,0.1,0.2])/num_valid_images*100)

%%
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
