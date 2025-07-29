%% whole pipeline (gt)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/gt_semantics/"+dataset_ids(i)+"_full_record_2.mat";
    data_ercord = load(file_name);
    time_CM=[time_CM;data_ercord.Record_CM.time_rot+data_ercord.Record_CM.time_t];
    time_SCM_trunc=[time_SCM_trunc;data_ercord.Record_SCM_trunc.time_rot+data_ercord.Record_SCM_trunc.time_t];
    time_SCM_ML=[time_SCM_ML;data_ercord.Record_trans_SCM_ML_list{5}.time_rot+data_ercord.Record_trans_SCM_ML_list{5}.time_t];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))
%% whole pipeline (pred)
dataset_ids = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
time_CM=[];
time_SCM_trunc=[];
time_SCM_ML=[];
for i=1:4
    file_name="./matlab/Experiments/records/pred_semantics/"+dataset_ids(i)+"_pred_full_record_2.mat";
    data_ercord = load(file_name);
    time_CM=[time_CM;data_ercord.Record_CM.time_rot+data_ercord.Record_CM.time_t];
    time_SCM_trunc=[time_SCM_trunc;data_ercord.Record_SCM_trunc.time_rot+data_ercord.Record_SCM_trunc.time_t];
    time_SCM_ML=[time_SCM_ML;data_ercord.Record_trans_SCM_ML_list{3}.time_rot+data_ercord.Record_trans_SCM_ML_list{3}.time_t];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_trunc: %f,%f,%f\n",quantile(time_SCM_trunc,[0.25,0.5,0.75]))
fprintf("SCM_FGO_ML: %f,%f,%f\n",quantile(time_SCM_ML,[0.25,0.5,0.75]))