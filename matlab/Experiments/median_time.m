%% rot
dataset_ids = ["689fec23d7","c173f62b15","69e5939669","a1d9da703c"];
time_CM=[];
time_SCM_power=[];
time_SCM_exp=[];
time_SCM_entropy=[];
for i=1:4
    file_name="./matlab/Experiments/records/"+dataset_ids(i)+"_rotation_record.mat";
    rot_ercord = load(file_name);
    time_CM=[time_CM;rot_ercord.Record_CM_FGO.Time];
    time_SCM_power=[time_SCM_power;rot_ercord.Record_SCM_FGO_power.Time];
    time_SCM_exp=[time_SCM_exp;rot_ercord.Record_SCM_FGO_exp.Time];
    time_SCM_entropy=[time_SCM_entropy;rot_ercord.Record_SCM_FGO_entropy.Time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_power: %f,%f,%f\n",quantile(time_SCM_power,[0.25,0.5,0.75]))
fprintf("SCM_FGO_exp: %f,%f,%f\n",quantile(time_SCM_exp,[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(time_SCM_entropy,[0.25,0.5,0.75]))
%% trans
dataset_ids = ["69e5939669","c173f62b15","689fec23d7","a1d9da703c"];
time_CM=[];
time_SCM_power=[];
time_SCM_exp=[];
time_SCM_entropy=[];
for i=1:4
    file_name="./matlab/Experiments/records/"+dataset_ids(i)+"_translation_record.mat";
    rot_ercord = load(file_name);
    time_CM=[time_CM;rot_ercord.Record_gt_CM.time];
    time_SCM_power=[time_SCM_power;rot_ercord.Record_gt_SCM_power.time];
    time_SCM_exp=[time_SCM_exp;rot_ercord.Record_gt_SCM_exp.time];
    time_SCM_entropy=[time_SCM_entropy;rot_ercord.Record_gt_SCM_entropy.time];
end
fprintf("CM_FGO: %f,%f,%f\n",quantile(time_CM,[0.25,0.5,0.75]))
fprintf("SCM_FGO_power: %f,%f,%f\n",quantile(time_SCM_power,[0.25,0.5,0.75]))
fprintf("SCM_FGO_exp: %f,%f,%f\n",quantile(time_SCM_exp,[0.25,0.5,0.75]))
fprintf("SCM_FGO_entropy: %f,%f,%f\n",quantile(time_SCM_entropy,[0.25,0.5,0.75]))
%% whole_pipeline
dataset_ids = ["69e5939669","c173f62b15","689fec23d7","a1d9da703c"];
time_SCM_entropy=[];
for i=1:4
    file_name="./matlab/Experiments/records/"+dataset_ids(i)+"_full_record.mat";
    rot_ercord = load(file_name);
    time_SCM_entropy=[time_SCM_entropy;rot_ercord.Record_est_SCM_entropy.time];
end
fprintf("time_SCM_entropy: %f,%f,%f\n",quantile(time_SCM_entropy,[0.25,0.5,0.75]))