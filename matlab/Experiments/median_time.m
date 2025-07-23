%% rot (gt)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/gt_semantics/"+dataset_ids(i)+"_rotation_record.mat";
    rot_ercord = load(file_name);
    time_CM=[time_CM;rot_ercord.Record_CM_FGO.Time];
    time_SCM_trunc=[time_SCM_trunc;rot_ercord.Record_SCM_trunc.Time];
    time_SCM_ML=[time_SCM_ML;rot_ercord.Record_SCM_ML_lists{2}.Time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))
%% rot (pred)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/pred_semantics/"+dataset_ids(i)+"_pred_rotation_record.mat";
    rot_ercord = load(file_name);
    time_CM=[time_CM;rot_ercord.Record_CM_FGO.Time];
    time_SCM_trunc=[time_SCM_trunc;rot_ercord.Record_SCM_trunc.Time];
    time_SCM_ML=[time_SCM_ML;rot_ercord.Record_SCM_ML_lists{2}.Time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))

%% trans(gt)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/gt_semantics/"+dataset_ids(i)+"_translation_record.mat";
    trans_ercord = load(file_name);
    time_CM=[time_CM;trans_ercord.Record_gt_CM.time];
    time_SCM_trunc=[time_SCM_trunc;trans_ercord.Record_gt_SCM_trunc.time];
    time_SCM_ML=[time_SCM_ML;trans_ercord.Record_SCM_ML_lists{2}.time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))

%% trans(pred)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/pred_semantics/"+dataset_ids(i)+"_pred_translation_record.mat";
    trans_ercord = load(file_name);
    time_CM=[time_CM;trans_ercord.Record_gt_CM.time];
    time_SCM_trunc=[time_SCM_trunc;trans_ercord.Record_gt_SCM_trunc.time];
    time_SCM_ML=[time_SCM_ML;trans_ercord.Record_SCM_ML_lists{2}.time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))