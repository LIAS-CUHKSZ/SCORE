%%% Rotation Recall w.r.t parameter q (gt semantics)
dataset_names = ["S1","S2","S3","S4"];
scene_name = ["S1 workstation","S2 office","S3 game bar","S4 art room"];
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
    color_palette=["#fbd0a6","#f37022","#b11016","#2aba9e","#007096"];
    figure
    for k=1:num_q
        plot(error_tick,Recall_buffer_2(k,:),"LineWidth",linewidth+0.5,"Color",color_palette(k))
        hold on
    end
    for k=1:num_q
        plot(error_tick,Recall_buffer_8(k,:),"LineWidth",linewidth,"Color",color_palette(k),"LineStyle","--")
        hold on
    end
    lgd = legend(["$q=0.6~(\pi)$","$q=0.7~(\pi)$","$q=0.8~(\pi)$","$q=0.9~(\pi)$","$q=0.99~(\pi)$",...
            "$q=0.6~(\pi/2)$","$q=0.7~(\pi/2)$","$q=0.8~(\pi/2)$","$q=0.9~(\pi/2)$","$q=0.99~(\pi/2)$"],"Interpreter","latex",...
            "FontName","Times New Roman","FontSize",12,"Location","southeast");
    lgd.NumColumns = 2;
    xlabel("Rotation Estimation Error (\circ)")
    ylabel("Recall")
    saveas(gcf,"./matlab/Experiments/records/sensitivity_S"+num2str(idx)+"_gt.pdf")
end

%%
%%% Rotation Recall w.r.t parameter q (predicted semantics)
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
    color_palette=["#fbd0a6","#f37022","#b11016","#2aba9e","#007096"];
    figure
    for k=1:num_q
        plot(error_tick,Recall_buffer_2(k,:),"LineWidth",linewidth+0.5,"Color",color_palette(k))
        hold on
    end
    for k=1:num_q
        plot(error_tick,Recall_buffer_8(k,:),"LineWidth",linewidth,"Color",color_palette(k),'LineStyle','--')
        hold on
    end
    lgd = legend(["$q=0.2~(\pi)$","$q=0.35~(\pi)$","$q=0.5~(\pi)$","$q=0.65~(\pi)$","$q=0.8~(\pi)$",...
            "$q=0.2~(\pi/2)$","$q=0.35~(\pi/2)$","$q=0.5~(\pi/2)$","$q=0.65~(\pi/2)$","$q=0.8~(\pi/2)$"],"Interpreter","latex",...
            "FontName","Times New Roman","FontSize",12,"Location","southeast");
    lgd.NumColumns = 2;
    xlabel("Rotation Estimation Error (\circ)")
    ylabel("Recall")
    saveas(gcf,"./matlab/Experiments/records/Sensitivity_S"+num2str(idx)+"_pred.pdf")
end

%%
%%% Translation Recall w.r.t parameter q (gt semantics)
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
scene_name = ["S1 workstation","S2 office","S3 game bar","S4 art room"];
for idx = 1:4
    dataset_name = dataset_names(idx);
    data_name_2     ="./matlab/Experiments/records/gt_semantics/"+dataset_name+"_full_record_2.mat";
    Record_data_2 = load(data_name_2);
    data_name_8     ="./matlab/Experiments/records/gt_semantics/"+dataset_name+"_full_record_8.mat";
    Record_data_8 = load(data_name_8);
    %
    q_list = Record_data_2.q_list;
    num_q  = 5;
    image_num = height(Record_data_2.Record_CM);
    %
    Record_ML_list_2 = Record_data_2.Record_trans_SCM_ML_list;
    Error_buffer_2 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_2{k};
        Error_buffer_2(:,k)=record_k.("Trans Err");
    end
    %
    Record_ML_list_8 = Record_data_8.Record_trans_SCM_ML_list;
    Error_buffer_8 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_8{k};
        Error_buffer_8(:,k)=record_k.("Trans Err");
    end
    %
    error_tick = 0.025:0.025:1;
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
    color_palette=["#fbd0a6","#f37022","#b11016","#2aba9e","#007096"];
    figure
    for k=1:num_q
        plot(error_tick,Recall_buffer_2(k,:),"LineWidth",linewidth+0.5,"Color",color_palette(k))
        hold on
    end
    for k=1:num_q
        plot(error_tick,Recall_buffer_8(k,:),"LineWidth",linewidth,"Color",color_palette(k),'LineStyle','--')
        hold on
    end
    lgd = legend(["$q=0.3~(\pi)$","$q=0.5~(\pi)$","$q=0.7~(\pi)$","$q=0.9~(\pi)$","$q=0.99~(\pi)$",...
            "$q=0.3~(\pi/2)$","$q=0.5~(\pi/2)$","$q=0.7~(\pi/2)$","$q=0.9~(\pi/2)$","$q=0.99~(\pi/2)$"],"Interpreter","latex",...
            "FontName","Times New Roman","FontSize",12,"Location","southeast");
    lgd.NumColumns = 2;
    xlabel("Translation Estimation Error (m)")
    ylabel("Recall")
    saveas(gcf,"./matlab/Experiments/records/Sensitivity_S"+num2str(idx)+"_trans_gt.pdf")
end
%%
%%% Translation Recall w.r.t parameter q (predicted semantics)
dataset_names = ["a1d9da703c","689fec23d7","c173f62b15","69e5939669"];
scene_name = ["S1 workstation","S2 office","S3 game bar","S4 art room"];
for idx = 1:4
    dataset_name = dataset_names(idx);
    data_name_2     ="./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_full_record_2.mat";
    Record_data_2 = load(data_name_2);
    data_name_8     ="./matlab/Experiments/records/pred_semantics/"+dataset_name+"_pred_full_record_8.mat";
    Record_data_8 = load(data_name_8);
    %
    q_list = Record_data_2.q_list;
    num_q  = 5;
    image_num = height(Record_data_2.Record_CM);
    %
    Record_ML_list_2 = Record_data_2.Record_trans_SCM_ML_list;
    Error_buffer_2 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_2{k};
        Error_buffer_2(:,k)=record_k.("Trans Err");
    end
    %
    Record_ML_list_8 = Record_data_8.Record_trans_SCM_ML_list;
    Error_buffer_8 = zeros(image_num,num_q);
    for k = 1:num_q
        record_k = Record_ML_list_8{k};
        Error_buffer_8(:,k)=record_k.("Trans Err");
    end
    %
    error_tick = 0.025:0.025:1;
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
    color_palette=["#fbd0a6","#f37022","#b11016","#2aba9e","#007096"];
    figure
    for k=1:num_q
        plot(error_tick,Recall_buffer_2(k,:),"LineWidth",linewidth+0.5,"Color",color_palette(k))
        hold on
    end
    for k=1:num_q
        plot(error_tick,Recall_buffer_8(k,:),"LineWidth",linewidth,"Color",color_palette(k),'LineStyle','--')
        hold on
    end
    lgd = legend(["$q=0.3~(\pi)$","$q=0.5~(\pi)$","$q=0.7~(\pi)$","$q=0.9~(\pi)$","$q=0.99~(\pi)$",...
            "$q=0.3~(\pi/2)$","$q=0.5~(\pi/2)$","$q=0.7~(\pi/2)$","$q=0.9~(\pi/2)$","$q=0.99~(\pi/2)$"],"Interpreter","latex",...
            "FontName","Times New Roman","FontSize",12,"Location","southeast");
    lgd.NumColumns = 2;
    xlabel("Translation Estimation Error (m)")
    ylabel("Recall")
    saveas(gcf,"./matlab/Experiments/records/Sensitivity_S"+num2str(idx)+"_trans_pred.pdf")
end
