% Save 3D line data as .mat file.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT
%%%%
%%
clear;
clc;
%%% config
% dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_ids = ["69e5939669"];
%%%
for n=1:length(dataset_ids)
    dataset_id = dataset_ids(n);
    datasetname="csv_dataset/"+dataset_id;
    % Nx7: endpoint_a, endpoint_b, semantic label
    lines3D=readmatrix(datasetname+"/3Dlines.csv");                     
    output_name =datasetname+ "/lines3D.mat";
    save(output_name,"lines3D");
end

