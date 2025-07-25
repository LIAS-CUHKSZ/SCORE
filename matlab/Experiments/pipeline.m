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
pred_flag = 0;
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
    rot_data_path = "./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record.mat";
    remapping = load(data_folder+"remapping.txt");
    rot_k_idx = 2;
else
    data_folder="csv_dataset/"+dataset_name+"/";
    rot_data_path = "./matlab/Experiments/records/gt_semantics/"+dataset_name+"_rotation_record.mat";
    remapping=[];
    rot_k_idx = 5;
end
lines3D=readmatrix(data_folder+"/3Dlines.csv");
%%% read rota data
rot_data = load(rot_data_path);
record_rot_SCM_ML = rot_data.Record_SCM_ML_lists{rot_k_idx};
epsilon_r = 0.015;
%%% trans params
branch_reso_t = 0.01; % terminate bnb when branch size <= branch_reso
prox_thres_t  = 0.01; %
epsilon_t = 0.03;
L_list = [3,5,10,30,50,100,300];
num_q = length(L_list);
%%% statistics
column_names=["Image Id","# 2D lines","Outlier Ratio","IR Err Rot","IR Err Trans", "Rot Err","Trans Err", "time"];
columnTypes = ["int32"  ,"int32"     ,"double"       ,"double"    ,"double",       "double" ,"double"   ,"double"];
valid_num = height(record_rot_SCM_ML);
temp_buffer = cell(valid_num,num_q);
%%
parfor num =1:valid_num
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_idx=record_rot_SCM_ML.("Image ID")(num);
    frame_id = sprintf("%06d",img_idx);
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
    fprintf(num2str(img_idx)+"\n")
    for k = 1:num_q
        time_all = 0;
        % saturation function design
        time_all = time_all + time_rot;
        %
        best_t_score = -1; best_R = eye(3); best_t = zeros(3,1);
        for n = 1:num_candidate_rot
            R_opt = Rot_candidates(n*3-2:n*3,:);
            [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot] = ...
                preprocess_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r);
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
            kernel_buff_SCM_ML = zeros(num_2D_lines,max(match_count_pruned));
            for i = 1:num_2D_lines
                if match_count_pruned(i)==0
                    continue
                end
                for j =1:match_count_pruned(i)
                    kernel_buff_SCM_ML(i,j) = log(1+L_list(k)*j/match_count_pruned(i))-log(1+L_list(k)*(j-1)/match_count_pruned(i));
                end
            end
            %%% Sat-CM: entropy + geo
            [t_best_candidates,~,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_ML,space_size,branch_reso_t,epsilon_t,prox_thres_t);
            time_all = time_all+time;
            % prune candidates according to geometric constraints
            [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM_ML);
            t_fine_tuned = tune_t(t_best_candidates,pert_rot_n_2D_inlier,endpoints_3D_inlier(1:2:end,:),epsilon_t);
            if best_score > best_t_score
                best_t_score = best_score;
                best_R = R_opt;
                best_t = t_fine_tuned;
            end
        end
        rot_err = angular_distance(best_R,R_gt);
        t_err   = norm(best_t-t_gt);
        temp_buffer{num,k}={img_idx,num_2D_lines,outlier_ratio,IR_err_rot,IR_err_trans,rot_err,t_err,time_all};
    end
end
%%
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

if pred_flag
    output_filename= "./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_full_record.mat";
else
    output_filename= "./matlab/Experiments/records/gt_semantics/"+dataset_name+"_full_record.mat";
end
save(output_filename);
%%
num_valid_f = height(Record_trans_SCM_ML_list{1});
const = epsilon_t/1;
fprintf("============ rot err quantile ============\n")
for k = 1:num_q
    this_record = Record_trans_SCM_ML_list{k};
    fprintf("SCM_FGO_ML(q=%f,L=%f): %f,%f,%f\n",const*L_list(k)/(1+const*L_list(k)),L_list(k),quantile(this_record.("Rot Err"),[0.25,0.5,0.75]))
end
fprintf("============ trans err quantile ============\n")
for k =1:num_q
    fprintf("SCM_FGO_entropy(q=%f,L=%f): %f,%f,%f\n",const*L_list(k)/(1+const*L_list(k)),L_list(k),quantile(this_record.("Trans Err"),[0.25,0.5,0.75]))
end
fprintf("============ Recall at 3/5/10 deg ============\n")
for k = 1:num_q
    fprintf("SCM_FGO_entropy(q=%f,L=%f): %f,%f,%f\n",const*L_list(k)/(1+const*L_list(k)),L_list(k),sum(this_record.("Rot Err")<[3,5,10])/num_valid_f*100)
end
fprintf("============ Recall at 5cm/10cm/20cm ============\n")
for k = 1:num_q
    fprintf("SCM_FGO_entropy(q=%f,L=%f): %f,%f,%f\n",const*L_list(k)/(1+const*L_list(k)),L_list(k),sum(this_record.("Trans Err")<[0.05,0.1,0.2])/num_valid_f*100)
end
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
    preprocess_rot(ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r)

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
