%% gt
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];     
time_rot_CM=[];
time_SCM_trunc=[];
time_rot_SCM_trunc=[];
time_SCM_ML=[];
time_rot_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/gt_semantics/"+dataset_ids(i)+"_full_record_8.mat";
    data_ercord = load(file_name);
    time_CM=[time_CM;data_ercord.Record_CM.time_rot+data_ercord.Record_CM.time_t];
    time_rot_CM = [time_rot_CM;data_ercord.Record_CM.time_rot];
    time_SCM_trunc=[time_SCM_trunc;data_ercord.Record_SCM_trunc.time_rot+data_ercord.Record_SCM_trunc.time_t];
    time_rot_SCM_trunc = [time_rot_SCM_trunc;data_ercord.Record_SCM_trunc.time_rot];
    time_SCM_ML=[time_SCM_ML;data_ercord.Record_trans_SCM_ML_list{5}.time_rot+data_ercord.Record_trans_SCM_ML_list{5}.time_t];
    time_rot_SCM_ML = [time_rot_SCM_ML;data_ercord.Record_trans_SCM_ML_list{5}.time_rot];
end
fprintf("========rotation time=========\n")
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_rot_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_rot_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_rot_SCM_ML,[0.25,0.5,0.75]))
fprintf("========overall time=========\n")
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))

%% pred
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_rot_CM=[];
time_SCM_trunc=[];
time_rot_SCM_trunc=[];
time_SCM_ML=[];
time_rot_SCM_ML=[];     
for i=1:4
    file_name="./matlab/backup/record_20IR_512/pred_semantics/"+dataset_ids(i)+"_pred_full_record_2.mat";
    % file_name="./matlab/Experiments/records/pred_semantics/"+dataset_ids(i)+"_pred_full_record_2.mat";
    data_ercord = load(file_name);
    time_CM=[time_CM;data_ercord.Record_CM.time_rot+data_ercord.Record_CM.time_t];
    time_rot_CM=[time_rot_CM;data_ercord.Record_CM.time_rot];
    time_SCM_trunc=[time_SCM_trunc;data_ercord.Record_SCM_trunc.time_rot+data_ercord.Record_SCM_trunc.time_t];
    time_rot_SCM_trunc = [time_rot_SCM_trunc;data_ercord.Record_SCM_trunc.time_rot];
    time_SCM_ML=[time_SCM_ML;data_ercord.Record_trans_SCM_ML_list{3}.time_rot+data_ercord.Record_trans_SCM_ML_list{3}.time_t];
    time_rot_SCM_ML = [time_rot_SCM_ML;data_ercord.Record_trans_SCM_ML_list{3}.time_rot];
end
fprintf("========rotation time=========\n")
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_rot_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_rot_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_rot_SCM_ML,[0.25,0.5,0.75]))
fprintf("========overall time=========\n")
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))



%% rotation (gt)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records_512_256/gt_semantics/"+dataset_ids(i)+"_rotation_record_8.mat";
    data_ercord = load(file_name);
    time_CM=[time_CM;data_ercord.Record_CM_FGO.Time];
    time_SCM_trunc=[time_SCM_trunc;data_ercord.Record_SCM_trunc.Time];
    time_SCM_ML=[time_SCM_ML;data_ercord.Record_SCM_ML_lists{4}.Time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))

%% rotation (pred)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/pred_semantics/"+dataset_ids(i)+"_pred_rotation_record_8.mat";
    data_ercord = load(file_name);
    time_CM=[time_CM;data_ercord.Record_CM_FGO.Time];
    time_SCM_trunc=[time_SCM_trunc;data_ercord.Record_SCM_trunc.Time];
    time_SCM_ML=[time_SCM_ML;data_ercord.Record_SCM_ML_lists{4}.Time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))