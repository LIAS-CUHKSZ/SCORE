%%%%
% Translation Estimation
% Saturated Consensus Maximization vs Consensus Maximization
% GT rotation vs FGO_PnL rotation estimates

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT

clear
clc
room_sizes = [10.3, 6, 2.6;  
              5.8,  4, 3.9; 
              10.4, 5, 3.3; 
              7,    7, 2.9];
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
scene_idx = 1;
dataset_name = dataset_ids(scene_idx);
space_size =  room_sizes(scene_idx,:);
data_folder="csv_dataset/"+dataset_name+"/";
load(data_folder+"lines3D.mat");
%%% params
branch_reso = 0.03; % terminate bnb when branch size <= branch_reso
prox_thres  = 0.01; %  
%%% statistics
column_names=["Image Id","# 2D lines","epsilon_t","Outlier Ratio","Rot Err","Max Trans Err","Min Trans Err","# Trans Candidates","BnB Score","GT Score","time"];
columnTypes = ["int32","int32","double","double","double","double","double","int32","double","double","double"];
total_img=2000;
Record_gt_CM = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_est_CM = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_gt_SCM = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_est_SCM = table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
%%% rotation data
Record_SCM_FGO = load("matlab\Experiments\records\"+dataset_name+"_rotation_record.mat").("Record_SCM_FGO");
epsilon_rs = Record_SCM_FGO.epsilon_r;
valid_idx = Record_SCM_FGO{:,1}; % These images have passed the rotation ambiguity test.
%% 
parfor num =1:length(valid_idx)
    % ---------------------------------------------------------------------
    % --- 1. load data ---
    img_idx=valid_idx(num);     
    frame_id = sprintf("%06d",img_idx);
    K_p=readmatrix(data_folder+"intrinsics\frame_"+frame_id+".csv");
    intrinsic=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(data_folder+"poses\frame_"+frame_id+".csv"); R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    retrived_3D_line_idx = readmatrix(data_folder+"retrived_3D_line_idx/frame_"+frame_id+".csv")+1;
    % lines2D(Nx9): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1) 
    lines2D = readmatrix(data_folder+"lines2D\frame_"+frame_id+"_2Dlines.csv"); 
    lines2D = lines2D(lines2D(:,4)~=0,:);   % delete 2D line without a semantic label
    lines2D(:,1:3)=lines2D(:,1:3)*intrinsic;  lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    num_2D_lines = size(lines2D,1);
    M = nnz(lines2D(:,9)>0);
    % ---------------------------------------------------------------------
    % --- 2. check observability ---
    ambiguiFlag = checkTransAmbiguity(img_idx,lines2D,R_gt);
    if ambiguiFlag
        continue
    end
    fprintf("Image"+num2str(img_idx)+"\n")
    % ---------------------------------------------------------------------
    % --- 3. semantic matching and saturation function design ---
    lines3D_sub = lines3D(retrived_3D_line_idx,:); % retrived sub-map
    [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D_sub);  
    epsilon_r = epsilon_rs(num);
    epsilon_t = max(lines2D(:,11))+0.001;
    
    % ---------------------------------------------------------------------
    % --- 4. localization with ground truth rotation ---
    R_opt = R_gt;
    [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,score_t_gt_CM,kernel_buff_SCM,score_t_gt_SCM] = ...
        under_specific_rot(num_2D_lines,ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r,t_gt,epsilon_t,intrinsic);
    kernel_buff_SCM_trunc = kernel_buff_SCM;
    kernel_buff_SCM_trunc(:,2:end)=0;
    %%% CM    
    [t_best_candidates,best_score,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,space_size,branch_reso,epsilon_t,prox_thres);
    % prune candidates according to physical constraints
    [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_CM);
    if num_candidate ~= size(t_best_candidates,2)
       fprintf("%d candidates pruned to %d \n",num_candidate,size(t_best_candidates,2));
       num_candidate = size(t_best_candidates,2);
    end
    [min_err,max_err,t_min]=min_max_trans_error(num_candidate,t_best_candidates,t_gt);
    Record_gt_CM(num+1,:)={img_idx,M,epsilon_t,1-M/size(pert_rot_n_2D_inlier,1),0,max_err,min_err,num_candidate,best_score,score_t_gt_CM,time};
    %%% Sat-CM 
    [t_best_candidates,best_score,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_trunc,space_size,branch_reso,epsilon_t,prox_thres);
    % prune candidates according to physical constraints
    [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM);
    if num_candidate ~= size(t_best_candidates,2)
       fprintf("%d candidates pruned to %d \n",num_candidate,size(t_best_candidates,2));
       num_candidate = size(t_best_candidates,2);
    end
    [min_err,max_err,t_min]=min_max_trans_error(num_candidate,t_best_candidates,t_gt);
    Record_gt_SCM(num+1,:)={img_idx,M,epsilon_t,1-M/size(pert_rot_n_2D_inlier,1),0,max_err,min_err,num_candidate,best_score,score_t_gt_SCM,time};
    
    % % ---------------------------------------------------------------------
    % % --- 5. localization with estimated rotation ---
    % rot_vec = Record_SCM_FGO.("Rot Vec"){num,:};
    % R_opt = rotvec2mat3d(rot_vec);
    % [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,score_t_gt_CM,kernel_buff_SCM,score_t_gt_SCM] = ...
    %     under_specific_rot(num_2D_lines,ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r,t_gt,epsilon_t,intrinsic);
    % kernel_buff_SCM_trunc = kernel_buff_SCM;
    % kernel_buff_SCM_trunc(:,2:end)=0;
    % %%% CM
    % [t_best_candidates,best_score,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,space_size,branch_reso,epsilon_t,prox_thres);
    % % prune candidates according to physical constraints
    % [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_CM);
    % if num_candidate ~= size(t_best_candidates,2)
    %    fprintf("%d candidates pruned to %d \n",num_candidate,size(t_best_candidates,2));
    %    num_candidate = size(t_best_candidates,2);
    % end
    % [min_err,max_err,t_min]=min_max_trans_error(num_candidate,t_best_candidates,t_gt);
    % Record_est_CM(num+1,:)={img_idx,M,epsilon_t,1-M/size(pert_rot_n_2D_inlier,1),0,max_err,min_err,num_candidate,best_score,score_t_gt_CM,time};
    % %%% SCM
    % [t_best_candidates,best_score,num_candidate,time,~,~] = Sat_TransFGO(pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_SCM_trunc,space_size,branch_reso,epsilon_t,prox_thres);
    % % prune candidates according to physical constraints
    % [best_score,t_best_candidates] = prune_t_candidates(R_opt,intrinsic,pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,epsilon_t,t_best_candidates,kernel_buff_SCM);
    % if num_candidate ~= size(t_best_candidates,2)
    %    fprintf("%d candidates pruned to %d \n",num_candidate,size(t_best_candidates,2));
    %    num_candidate = size(t_best_candidates,2);
    % end
    % [min_err,max_err,t_min]=min_max_trans_error(num_candidate,t_best_candidates,t_gt);
    % Record_est_SCM(num+1,:)={img_idx,M,epsilon_t,1-M/size(pert_rot_n_2D_inlier,1),0,max_err,min_err,num_candidate,best_score,score_t_gt_SCM,time};
end
Record_gt_CM(Record_gt_CM.("BnB Score")==0,:)=[];
Record_est_CM(Record_est_CM.("BnB Score")==0,:)=[];
Record_gt_SCM(Record_gt_SCM.("BnB Score")==0,:)=[];
Record_est_SCM(Record_est_SCM.("BnB Score")==0,:)=[];
%%
fprintf("============ statistics ============\n")
fprintf("num of valid images: %d\n",height(Record_gt_CM));
fprintf("num of re-localized images ( < 20 cm):\n")
fprintf("CM_gt_FGO: %d \n",length(find(Record_gt_CM.("Max Trans Err")<0.2)))
fprintf("CM_est_FGO: %d \n",length(find(Record_est_CM.("Max Trans Err")<0.2)))
fprintf("SCM_gt_FGO: %d \n",length(find(Record_gt_SCM.("Max Trans Err")<0.2)))
fprintf("SCM_est_FGO: %d \n",length(find(Record_est_SCM.("Max Trans Err")<0.2)))
output_filename= "./matlab/Experiments/records/"+dataset_name+"_translation_record.mat";
% save(output_filename);
% Record_gt_SCM(Record_gt_SCM.("# 2D lines")<Record_gt_SCM.("GT Score"),:)=[];

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


function [pert_rot_n_2D_inlier,endpoints_3D_inlier,id_inliers_under_rot,kernel_buff_CM,score_t_gt_CM,kernel_buff_SCM,score_t_gt_SCM] = ...
    under_specific_rot(num_2D_lines,ids,R_opt,v_3D,n_2D,endpoints_3D,epsilon_r,t_gt,epsilon_t,intrinsic)

    inlier_under_rot = find(abs(dot(R_opt'*v_3D',n_2D'))<=epsilon_r);
    id_inliers_under_rot = ids(inlier_under_rot);
    %%% saturation function design
    match_count = zeros(num_2D_lines,1);
    for i = 1:num_2D_lines
        match_count(i) = sum(id_inliers_under_rot==i);
    end
    kernel_buff_SCM = zeros(num_2D_lines,max(match_count));
    kernel_buff_CM  = ones(num_2D_lines,max(match_count));
    for i = 1:num_2D_lines
        kernel_buff_SCM(i,1)=1;
        for j = 2:match_count(i)
            kernel_buff_SCM(i,j)=log(j)-log(j-1);
        end
    end
    kernel_buff_SCM(:,2:end)=kernel_buff_SCM(:,2:end)/sum(sum(kernel_buff_SCM(:,2:end)));
    %%%
    n_2D_inlier=n_2D(inlier_under_rot,:); v_3D_inlier=v_3D(inlier_under_rot,:);
    endpoints_3D_inlier=endpoints_3D( sort( [ inlier_under_rot*2, inlier_under_rot*2-1 ] ), :);
    %%% fine tune n_2D_inlier, let it perfectly orthgonal to v_3D_inlier after rotation
    pert_rot_n_2D_inlier = pert_n((R_opt*n_2D_inlier')',v_3D_inlier);
    residuals = sum(pert_rot_n_2D_inlier.*(endpoints_3D_inlier(1:2:end,:)-t_gt'),2);
    inliers_t_gt = find(abs(residuals )<=epsilon_t);
    inliers_t_gt = prune_inliers(R_opt,intrinsic,inliers_t_gt,endpoints_3D_inlier,t_gt);
    score_t_gt_SCM = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_SCM);
    score_t_gt_CM = calculate_score(id_inliers_under_rot(inliers_t_gt),kernel_buff_CM);
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
