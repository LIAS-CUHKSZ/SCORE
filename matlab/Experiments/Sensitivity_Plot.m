%%% Rotation Recall w.r.t parameter q
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
scene_name = ["S1 workstation","S2 office","S3 game bar","S4 art room"];
% data_name_pred="./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record.mat";
% Record_data_pred = load(data_name_pred);
for idx = 1:4
    dataset_name = dataset_names(idx);
    data_name_2     ="./matlab/Experiments/records/gt_semantics/"+dataset_name+"_rotation_record_2.mat";
    Record_data_2 = load(data_name_2);
    data_name_8     ="./matlab/Experiments/records/gt_semantics/"+dataset_name+"_rotation_record_8.mat";
    Record_data_8 = load(data_name_8);
    %
    L_list = Record_data_2.L_list;
    num_q  = 5;
    image_num = height(Record_data_2.Record_CM_FGO);
    %
    Record_ML_list_2 = Record_data_2.Record_SCM_ML_lists;
    Error_buffer_2 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_2{k};
        Error_buffer_2(:,k)=record_k.("Max Rot Err");
    end
    %
    Record_ML_list_8 = Record_data_8.Record_SCM_ML_lists;
    Error_buffer_8 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_8{k};
        Error_buffer_8(:,k)=record_k.("Max Rot Err");
    end
    %
    error_tick = 0.5:0.5:10;
    Recall_buffer_2 = zeros(num_q,length(error_tick));
    Recall_buffer_8 = zeros(num_q,length(error_tick));
    for j = 1:length(error_tick)
        for k=1:num_q
            Recall_buffer_2(k,j) = nnz(Error_buffer_2(:,k)<=error_tick(j))/image_num;
            Recall_buffer_8(k,j) = nnz(Error_buffer_8(:,k)<=error_tick(j))/image_num;
        end
    end
    % plot
    linewidth=1.5;
    color_palette_2=["#f6e58d","#f9ca24","#f0932b","#eb4d4b","#b71540"];
    color_palette_8=["#A3C9F9","#6BB4F4","#2E9BEF","#1C87D7","#084B9E"];
    figure
    for k=1:num_q
        plot(error_tick,Recall_buffer_2(k,:),"LineWidth",linewidth,"Color",color_palette_2(k))
        hold on
    end
    for k=1:num_q
        plot(error_tick,Recall_buffer_8(k,:),"LineWidth",linewidth,"Color",color_palette_8(k))
        hold on
    end
    lgd = legend(["$L=66~(\pi)$","$L=66\sqrt{10}~(\pi)$","$L=660~(\pi)$","$L=660\sqrt{10}~(\pi)$","$L=6600~(\pi)$",...
            "$L=66~(\pi/2)$","$L=66\sqrt{10}~(\pi/2)$","$L=660~(\pi/2)$","$L=660\sqrt{10}~(\pi/2)$","$L=6600~(\pi/2)$"],"Interpreter","latex",...
            "FontName","Times New Roman","FontSize",12,"Location","southeast");
    lgd.NumColumns = 2;
    xlabel("Rotation Estimation Error (\circ)")
    ylabel("Recall")
    saveas(gcf,"./matlab/Experiments/records/sensitivity_S"+num2str(idx)+"_gt.pdf")
end



%%
%%% Rotation Recall w.r.t parameter q
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
scene_name = ["S1 workstation","S2 office","S3 game bar","S4 art room"];
for idx = 1:4
    dataset_name = dataset_names(idx);
    data_name_2     ="./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record_2.mat";
    Record_data_2 = load(data_name_2);
    data_name_8     ="./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_rotation_record_8.mat";
    Record_data_8 = load(data_name_8);
    %
    L_list = Record_data_2.L_list;
    num_q  = 5;
    image_num = height(Record_data_2.Record_CM_FGO);
    %
    Record_ML_list_2 = Record_data_2.Record_SCM_ML_lists;
    Error_buffer_2 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_2{k};
        Error_buffer_2(:,k)=record_k.("Max Rot Err");
    end
    %
    Record_ML_list_8 = Record_data_8.Record_SCM_ML_lists;
    Error_buffer_8 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_8{k};
        Error_buffer_8(:,k)=record_k.("Max Rot Err");
    end
    %
    error_tick = 0.5:0.5:10;
    Recall_buffer_2 = zeros(num_q,length(error_tick));
    Recall_buffer_8 = zeros(num_q,length(error_tick));
    for j = 1:length(error_tick)
        for k=1:num_q
            Recall_buffer_2(k,j) = nnz(Error_buffer_2(:,k)<=error_tick(j))/image_num;
            Recall_buffer_8(k,j) = nnz(Error_buffer_8(:,k)<=error_tick(j))/image_num;
        end
    end
    % plot
    linewidth=1.5;
    color_palette_2=["#f6e58d","#f9ca24","#f0932b","#eb4d4b","#b71540"];
    color_palette_8=["#A3C9F9","#6BB4F4","#2E9BEF","#1C87D7","#084B9E"];
    figure
    for k=1:num_q
        plot(error_tick,Recall_buffer_2(k,:),"LineWidth",linewidth,"Color",color_palette_2(k))
        hold on
    end
    for k=1:num_q
        plot(error_tick,Recall_buffer_8(k,:),"LineWidth",linewidth,"Color",color_palette_8(k))
        hold on
    end
    lgd = legend(["$L=30~(\pi)$","$L=60~(\pi)$","$L=120~(\pi)$","$L=240~(\pi)$","$L=360~(\pi)$",...
            "$L=30~(\pi/2)$","$L=60~(\pi/2)$","$L=120~(\pi/2)$","$L=240~(\pi/2)$","$L=360~(\pi/2)$"],"Interpreter","latex",...
            "FontName","Times New Roman","FontSize",12,"Location","southeast");
    lgd.NumColumns = 2;
    xlabel("Rotation Estimation Error (\circ)")
    ylabel("Recall")
    saveas(gcf,"./matlab/Experiments/records/Sensitivity_S"+num2str(idx)+"_pred.pdf")
end
