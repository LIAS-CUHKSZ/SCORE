%%% Rotation Recall w.r.t parameter q
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
scene_idx = 4; % choose one scene
pred_flag = 1; % set 1 if use predicted semantic label
dataset_name = dataset_names(scene_idx);
% load semantic remapping
if pred_flag
    data_name="./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record.mat";
else
    data_name="./matlab/Experiments/records/gt_semantics/"+dataset_name+"_rotation_record.mat";
end
Record_data = load(data_name);
q_list = Record_data.q_list;
num_q  = Record_data.num_q;
Record_ML_list = Record_data.Record_SCM_ML_lists;
image_num = height(Record_data.Record_CM_FGO);
Error_buffer = zeros(image_num,num_q);
for k = 1:num_q
    record_k = Record_ML_list{k};
    Error_buffer(:,k)=record_k.("Max Rot Err");
end
%%
error_tick = [1,2,3,5,10,30];
Recall_buffer = zeros(num_q,length(error_tick));

for j = 1:length(error_tick)
    for k=1:num_q
        Recall_buffer(k,j) = nnz(Error_buffer(:,k)<=error_tick(j))/image_num;
    end
end
plot(1)
for k=1:num_q
    plot(error_tick,Recall_buffer(k,:))
    hold on
end
legend(["q=0.5","q=0.6","q=0.7","q=0.8","q=0.9"])