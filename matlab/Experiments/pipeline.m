%%%%
% complete pipeline with Saturated Consensus Maximization
% --- Note!! ---
% If you don't want to or can't use the compiled mex functions,
% remeber to set variables 'mex_flag=0' in functions Sat_RotFGO and Sat_TransFGO
% this code runs on the rotation results from running rotation.m

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT

clear
clc
room_sizes =  [ 8,    6, 4;
    7,   7, 3;
    10.5, 5, 3.5;
    10.5, 6, 3.0];
dataset_names = ["S1","S2","S3","S4"];

% configure setting
scene_idx = 2; % which scene
dataset_name=dataset_names(scene_idx);
space_size =  room_sizes(scene_idx,:);
pred_flag = 0; % use predicted label?
two_or_eight = 0; % side length = pi or pi/2?

% set params
branch_reso_t = 0.02;          % terminate bnb when branch size <= branch_reso
prox_thres_t  = branch_reso_t; % cluster proximate stabbers
epsilon_t = 0.03;              % translation residual tolerance
u_t=1;                         % translation residual upper bound
q_list = [0.3,0.5,0.7,0.9,0.99];
C_list = q_list./(1-q_list)*u_t/epsilon_t;
num_q = length(C_list);

% read rotation data
if pred_flag
    data_folder="csv_dataset/"+dataset_name+"_pred/";
    if two_or_eight
        rot_data_path = "./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record_2.mat";
    else
        rot_data_path = "./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record_8.mat";
    end
    remapping = load(data_folder+"remapping.txt");
    rot_k_idx = 3; % choose q = 0.5
else
    data_folder="csv_dataset/"+dataset_name+"/";
    if two_or_eight
        rot_data_path = "./matlab/Experiments/records/gt_semantics/"+dataset_name+"_rotation_record_2.mat";
    else
        rot_data_path = "./matlab/Experiments/records/gt_semantics/"+dataset_name+"_rotation_record_8.mat";
    end
    remapping=[];
    rot_k_idx = 4; % choose q = 0.9
end
rot_data = load(rot_data_path);
record_rot_SCM_ML = rot_data.Record_SCM_ML_lists{rot_k_idx};
epsilon_r = 0.015;

% record table
column_names=["Image Name","# 2D lines","Outlier Ratio","IR Err Rot","IR Err Trans", "Rot Err","Trans Err","time_rot","time_t","best_R","best_t"];
columnTypes = ["string"  ,"int32"     ,"double"       ,"double"    ,"double",       "double" ,"double"   ,"double","double","cell","cell"];
valid_num = height(record_rot_SCM_ML);
Record_CM = table('Size', [valid_num, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_trunc = table('Size', [valid_num, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);

% go through each query image
temp_buffer = cell(valid_num,num_q);
img_names = record_rot_SCM_ML.("Image Name");
lines3D=readmatrix(data_folder+"/3Dlines.csv");

% parfor num =1:valid_num % use parfor to run in parallel
for num =1:valid_num
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_name = img_names(num);
    K_p=readmatrix(data_folder+"intrinsics/"+img_name+".csv");
    T_gt = readmatrix(data_folder+"poses/"+img_name+".csv");
    R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);

    % load image retrivel results and prune search space
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1]; % intrinsic matrix
    % lines2D(Nx11): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1), rot_err(1), trans_err(1)
    lines2D = readmatrix(data_folder+"lines2D/"+img_name+"_2Dlines.csv");
    lines2D(lines2D(:,4)==0,:)=[]; % delete lines without semantic label
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic; lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';

    % load image retrivel results and prune search space
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/"+img_name+".csv")+1; % retrived sub-map
    retrived_closest_pose = readmatrix(data_folder+"retrived_closest_pose/"+img_name+"_retrived_pose.csv"); % pose of the most similar retrived image
    [alpha,phi,theta] = rot2angle(retrived_closest_pose(1:3,1:3)');
    IR_err_rot = angular_distance(retrived_closest_pose(1:3,1:3),R_gt);
    IR_err_trans = norm(t_gt-retrived_closest_pose(1:3,4));
    lines3D_sub = lines3D(retrived_3D_line_idx,:);

    % ---------------------------------------------------------------------
    % --- 2. semantic matching ---
    [lines2D,lines3D_sub]=remap_semantic_id(lines2D,lines3D_sub,remapping);
    [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D_sub);  % match with clustered 3D lines
    num_2D_lines = size(lines2D,1);

    % --- 3. load rotation data ---
    Rot_candidates = record_rot_SCM_ML.("Rot Candidates"){num};
    num_candidate_rot = record_rot_SCM_ML.("# Rot Candidates")(num);
    outlier_ratio = record_rot_SCM_ML.("Outlier Ratio")(num);
    time_rot = record_rot_SCM_ML.Time(num);

    %-------------------------------------------------------------
    %---- 4. complete pipeline starts here -----
    fprintf(img_name+"\n")
    time_t_CM = 0; best_score_CM = -1; best_R_CM = eye(3); best_t_CM = zeros(3,1);
    time_t_trunc = 0; best_score_trunc = -1; best_R_trunc = eye(3); best_t_trunc = zeros(3,1);
    time_t_ML = zeros(num_q,1); best_score_ML = -ones(num_q,1); best_R_ML=cell(num_q,1); best_t_ML=cell(num_q,1);
    % go through all candidates
    for n = 1:num_candidate_rot
        R_opt = Rot_candidates(n*3-2:n*3,:);
        [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
            preprocess_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r);
        % observability check
        ambiguiFlag = checkTransAmbiguity(img_name,lines2D(unique(id_inliers_under_rot),:),R_opt);
        if ambiguiFlag
            continue
        end
        % saturation function design
        match_count_pruned = zeros(num_2D_lines,1);
        for i = 1:num_2D_lines
            match_count_pruned(i) = sum(id_inliers_under_rot==i);
        end
        sat_buff_CM     = ones(num_2D_lines,max(match_count_pruned));
        sat_buff_SCM_trunc = zeros(num_2D_lines,max(match_count_pruned));
        sat_buff_SCM_trunc(:,1) = 1;
        sat_buff_ML_list = cell(num_q,1);
        for k = 1: num_q
            sat_buff_ML_list{k} = zeros(num_2D_lines,max(match_count_pruned));
            for i = 1:num_2D_lines
                if match_count_pruned(i)==0
                    continue
                end
                for j =1:match_count_pruned(i)
                    sat_buff_ML_list{k}(i,j) = log(1+C_list(k)*j/match_count_pruned(i))-log(1+C_list(k)*(j-1)/match_count_pruned(i));
                end
            end
        end

        %%% CM
        [t_best_candidates,~,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,sat_buff_CM,space_size,branch_reso_t,epsilon_t,prox_thres_t);
        time_t_CM = time_t_CM+time;
        % prune candidates according to geometric constraints
        [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,sat_buff_CM);
        t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
        if best_score > best_score_CM
            best_score_CM = best_score;
            best_R_CM = R_opt;
            best_t_CM = t_fine_tuned;
        end

        %%% SCM trunc
        [t_best_candidates,~,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,sat_buff_SCM_trunc,space_size,branch_reso_t,epsilon_t,prox_thres_t);
        time_t_trunc = time_t_trunc+time;
        % prune candidates according to geometric constraints
        [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,sat_buff_SCM_trunc);
        t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
        if best_score > best_score_trunc
            best_score_trunc = best_score;
            best_R_trunc = R_opt;
            best_t_trunc = t_fine_tuned;
        end

        %%% SCM ML
        for k = 1:num_q
            this_ML_buffer = sat_buff_ML_list{k};
            [t_best_candidates,~,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,this_ML_buffer,space_size,branch_reso_t,epsilon_t,prox_thres_t);
            time_t_ML(k) = time_t_ML(k)+time;
            % prune candidates according to geometric constraints
            [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,this_ML_buffer);
            t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
            if best_score > best_score_ML(k)
                best_score_ML(k) = best_score;
                best_R_ML{k} = R_opt;
                best_t_ML{k} = t_fine_tuned;
            end
        end
    end
    %
    rot_err = angular_distance(best_R_CM,R_gt); t_err= norm(best_t_CM-t_gt);
    Record_CM(num,:) = {img_name,num_2D_lines,outlier_ratio,IR_err_rot,IR_err_trans,rot_err,t_err,time_rot,time_t_CM,best_R_CM,best_t_CM};
    %
    rot_err = angular_distance(best_R_trunc,R_gt); t_err= norm(best_t_trunc-t_gt);
    Record_SCM_trunc(num,:) = {img_name,num_2D_lines,outlier_ratio,IR_err_rot,IR_err_trans,rot_err,t_err,time_rot,time_t_trunc,best_R_trunc,best_t_trunc};
    %
    for k = 1:num_q
        rot_err = angular_distance(best_R_ML{k},R_gt); t_err= norm(best_t_ML{k}-t_gt);
        temp_buffer{num,k} = {img_name,num_2D_lines,outlier_ratio,IR_err_rot,IR_err_trans,rot_err,t_err,time_rot,time_t_ML(k),best_R_ML{k},best_t_ML{k}};
    end
end
% copy data from temp_buffer
Record_trans_SCM_ML_list = cell(num_q,1);
for k = 1:num_q
    Record_trans_SCM_ML_list{k} = table('Size', [valid_num, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
end
for num=1:valid_num
    for k=1:num_q
        temp_result = temp_buffer{num,k};
        if ~isempty(temp_result)
            Record_trans_SCM_ML_list{k}(num,:)=temp_result;
        end
    end
end
for k=1:num_q
    Record_trans_SCM_ML_list{k}(Record_trans_SCM_ML_list{k}.("Outlier Ratio")==0,:)=[];
end

%% output
if pred_flag
    output_folder = "./matlab/Experiments/records/pred_semantics/";
    if two_or_eight
        output_filename= output_folder+dataset_name+"_pred_full_record_2.mat";
    else
        output_filename= output_folder+dataset_name+"_pred_full_record_8.mat";
    end
else
    output_folder = "./matlab/Experiments/records/gt_semantics/";
    if two_or_eight
        output_filename= output_folder+dataset_name+"_full_record_2.mat";
    else
        output_filename= output_folder+dataset_name+"_full_record_8.mat";
    end
end
save(output_filename);

%%
% print statistics
num_valid_f = height(Record_trans_SCM_ML_list{1});
fprintf("============ rot err quantile ============\n")
fprintf("Image Retriveal: %f,%f,%f\n",quantile(Record_CM.("IR Err Rot"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(Record_CM.("Rot Err"),[0.25,0.5,0.75]))
fprintf("============ Recall at 3/5/10 deg ============\n")
fprintf("Image Retriveal: %f,%f,%f\n",sum(Record_CM.("IR Err Rot")<[3,5,10])/num_valid_f*100)
fprintf("SCM_FGO_ML: %f,%f,%f\n",sum(Record_CM.("Rot Err")<[3,5,10])/num_valid_f*100)
%
fprintf("============ trans err quantile ============\n")
fprintf("Image Retriveal: %f,%f,%f\n",quantile(Record_CM.("IR Err Trans"),[0.25,0.5,0.75]));
fprintf("CM_FGO: %f,%f,%f\n",quantile(Record_CM.("Trans Err"),[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(Record_SCM_trunc.("Trans Err"),[0.25,0.5,0.75]))
for k =1:num_q
    fprintf("SCM_FGO_ML(q=%f,C=%f): %f,%f,%f\n",q_list(k),C_list(k),quantile(Record_trans_SCM_ML_list{k}.("Trans Err"),[0.25,0.5,0.75]))
end
%
fprintf("============ Recall at 5cm/10cm/20cm ============\n")
fprintf("Image Retriveal: %f,%f,%f\n",sum(Record_CM.("IR Err Trans")<[0.05,0.1,0.2])/num_valid_f*100)
fprintf("CM_FGO: %f,%f,%f\n",sum(Record_CM.("Trans Err")<[0.05,0.1,0.2])/num_valid_f*100)
fprintf("SCM_FGO_trunc: %f,%f,%f\n",sum(Record_SCM_trunc.("Trans Err")<[0.05,0.1,0.2])/num_valid_f*100)
for k = 1:num_q
    fprintf("SCM_FGO_ML(q=%f,C=%f): %f,%f,%f\n",q_list(k),C_list(k),sum(Record_trans_SCM_ML_list{k}.("Trans Err")<[0.05,0.1,0.2])/num_valid_f*100)
end
%
fprintf("============ time quantile ============\n")
fprintf("Rot: %f,%f,%f\n",quantile(Record_CM.time_rot,[0.25,0.5,0.75]));
fprintf("Trans w CM_FGO:%f,%f,%f\n",quantile(Record_CM.time_t,[0.25,0.5,0.75]));
fprintf("Trans w SCM_trunc:%f,%f,%f\n",quantile(Record_SCM_trunc.time_t,[0.25,0.5,0.75]));
for k = 1:num_q
    fprintf("Trans w SCM_ML(q=%f,C=%f): %f,%f,%f\n",q_list(k),C_list(k),quantile(Record_trans_SCM_ML_list{k}.time_t,[0.25,0.5,0.75]));
end
%%
% ---------------------------------------------------------------------
% --- sub-functions ---
function flag = checkTransAmbiguity(img_name,lines2D,R_opt)
M = size(lines2D,1);
A_gt = zeros(M,3);
flag = false;
for i=1:M
    n = lines2D(i,1:3);
    A_gt(i,:)=(R_opt*n')';
end
if rank(A_gt'*A_gt)<3
    flag = true;
    fprintf(img_name+" is ambigious in translation, skip.\n");
end
end


function [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
    preprocess_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r)

inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
id_inliers_under_rot = ids(inlier_under_rot);
n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
%%% fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
end

function [best_score,t_real_candidate] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D,endpoints_3D,ids,epsilon_t,t_candidates,sat_buff)
p_3D = endpoints_3D(1:2:end,:);
best_score = -1; t_real_candidate=[];
for k=1:size(t_candidates,2)
    t_test = t_candidates(:,k);
    residuals = sum(pert_rot_n_2D.*(p_3D-t_test'),2);
    inliers = find(abs(residuals )<=epsilon_t);
    %%% the above inliers satisfy the geometric constraints,
    %%% we urther filter lines behind the camera and outside image
    real_inliers = prune_inliers(R_opt,intrinsic,inliers,endpoints_3D,t_test);
    score=calculate_score(ids(real_inliers),sat_buff);
    if score > best_score
        best_score = score; t_real_candidate = t_test;
    elseif score == best_score
        t_real_candidate = [t_real_candidate,t_test];
    else
    end
end
end