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
room_sizes = [10.3,6,2.6;  5.8, 4,3.9; 10.4 , 5 ,3.3; 7,7,2.9];
dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_idx = dataset_ids(3);
room_size = room_sizes(3,:);
data_folder="csv_dataset/"+dataset_idx+"/";
load(data_folder+"lines3D.mat");
%%% orientation params
kernel_SCM = @(x) x^(-9);  % as a close approximation to @(x) 1-(x>1)
trunc_num=length(lines3D);  kernel_buffer_SCM=zeros(trunc_num,1);
for i=1:trunc_num
    kernel_buffer_SCM(i)=kernel_SCM(i);
end
prox_thres = 1*pi/180; % keep candidates which have a same score and not proximate to each other
verbose_flag=0; % verbose mode for BnB 
mex_flag=1; % use matlab mex code for acceleration
branch_reso = pi/512; % terminate bnb when branch size <= branch_reso
sample_reso = pi/512; % resolution for interval analysis
total_img=1000;
%%% translation params
sampleSize=4; ransac_iterations=30000; 
%%
column_names=["Image Id","# 2D lines","epsilon_t","Outlier Ratio","Rot Err","Trans Err","# Trans Candidates","Ransac Score","GT Score",];
columnTypes = ["int32","int32","double","double","double","double","int32","double","double"];
Record_gt_ransac =table('Size', [total_img, length(column_names)],'VariableTypes', columnTypes,'VariableNames', column_names);
Record_SCM_FGO_clustered = load("matlab\Experiments\records\"+dataset_idx+"_rotation_record.mat").("Record_SCM_FGO_clustered");
epsilon_rs = Record_SCM_FGO_clustered.epsilon_r;
%%% Experiments
valid_idx = Record_SCM_FGO_clustered{:,1}; % These images have passed the rotation ambiguity test.
for num =1:length(valid_idx)
    epsilon_r = epsilon_rs(num);
    img_idx=valid_idx(num);     
    %%% read 2D line data of cur image
    frame_id = sprintf("%06d",img_idx);
    K_p=readmatrix(data_folder+"intrinsics\frame_"+frame_id+".csv");
    K=[K_p(1),0,K_p(2);0,K_p(3),K_p(4);0,0,1];
    T_gt = readmatrix(data_folder+"poses\frame_"+frame_id+".csv"); R_gt = T_gt(1:3,1:3); t_gt=T_gt(1:3,4);
    % lines2D(Nx9): normal vector(3x1), semantic label(1), endpoint a(u,v), endpoint b(u,v), matching 3d line idx(1) 
    lines2D = readmatrix(data_folder+"lines2D\frame_"+frame_id+"_2Dlines.csv"); 
    lines2D = lines2D(lines2D(:,4)~=0,:);   % delete 2D line without a semantic label
    %%%%% note: annotating the below line leads to a more challenging problem setting 
    lines2D = lines2D(lines2D(:,9)>=0,:); % delete 2D line without a real matching
    lines2D(:,1:3)=lines2D(:,1:3)*K;  lines2D(:,1:3)=lines2D(:,1:3)./vecnorm(lines2D(:,1:3)')';
    %%% check translation ambiguity
    M = size(lines2D,1);
    A_gt = zeros(M,3);
    for i=1:M
        n = lines2D(i,1:3); 
        A_gt(i,:)=(R_gt*n')';
    end
    if rank(A_gt'*A_gt)<3
        fprintf("image"+num2str(img_idx)+" is ambigious in translation, skip.\n");
        continue
    end
    %%% match 2D and 3D lines using semnatic label
    fprintf(num2str(img_idx)+"\n")
    [ids,n_2D,v_3D,endpoints_3D]=match_line(lines2D,lines3D);  % match with unclustered 3D lines 
    [ids_cluster,n_2D_cluster,v_3D_cluster,endpoints_3D_cluster]=match_line(lines2D,lines3D_cluster);  % match with clustered 3D lines
    epsilon_t = max(lines2D(:,11))+0.001;
    %%%%%%%%%%%%%%%%%%% Estimate Translation with GT Orient
    inlier_under_orient = find(abs(dot(R_gt'*v_3D',n_2D'))<=epsilon_r);
    id_inliers = ids(inlier_under_orient);
    if length(unique(id_inliers))<4
        continue
    end
    n_2D_inlier=n_2D(inlier_under_orient,:);
    v_3D_inlier=v_3D(inlier_under_orient,:);
    inlier_end_points = [inlier_under_orient*2,inlier_under_orient*2-1];
    endpoints_3D_inlier=endpoints_3D(sort(inlier_end_points),:);
    gt_inliers=[];
    for i=1:length(inlier_under_orient)
        rot_ni = R_gt*n_2D_inlier(i,:)';
        error_a = rot_ni'*(endpoints_3D_inlier(i*2-1,:)'-t_gt);
        error_b = rot_ni'*(endpoints_3D_inlier(i*2,:)'-t_gt);
        if sign(error_a)~=sign(error_b) || min(abs(error_a),abs(error_b))<=epsilon_t
            gt_inliers =[gt_inliers,i];
        end
    end
    gt_idx = id_inliers(gt_inliers);
    gt_score = calculate_score(gt_idx,@(x) x^(-9));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [t_opt,num_candidate,best_score] = sat_t_Ransac(n_2D_inlier',endpoints_3D_inlier,R_gt,id_inliers,sampleSize,ransac_iterations,epsilon_t,room_size);
    error_translation=-1;
    for k=1:num_candidate
        if error_translation<norm(t_opt(:,k)-t_gt)
            x_worst =t_opt(1,k);
            y_worst =t_opt(2,k);
            z_worst =t_opt(3,k);
            error_translation = norm(t_opt(:,k)-t_gt);
        end        
    end
    Record_gt_ransac(num+1,:)={img_idx,M,epsilon_t,1-M/size(n_2D_inlier,1),0,error_translation,num_candidate,best_score, gt_score};
end
Record_gt_ransac(Record_gt_ransac.("Ransac Score")==0,:)=[];