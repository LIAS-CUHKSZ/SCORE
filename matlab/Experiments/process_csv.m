
% Cluster parallel 3D lines.
% Save 3D line data as .mat file.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%
%%
clear;
clc;
%%% config
% dataset_ids = ["69e5939669","55b2bf8036","c173f62b15","689fec23d7"];
dataset_ids = ["55b2bf8036"];
cluster_degree = 0.5;
parallel_threshold=cosd(cluster_degree); % tune this if necessary
%%%
for n=1:length(dataset_ids)
    dataset_id = dataset_ids(n);
    datasetname="csv_dataset/"+dataset_id;
    lines3D=readmatrix(datasetname+"/3dlines.csv");                     % Nx13: p,v,semantic label, endpoint a, endpoint b
    % clustering parallel 3D lines with same semantics
    lines3D_cluster = lines3D;
    i=0;
    while(i<length(lines3D_cluster))
        i=i+1;
        vi=lines3D_cluster(i,4:6);
        parallel_idx=[];
        for j=i+1:length(lines3D_cluster)
            vj=lines3D_cluster(j,4:6);
            if ( abs(vi*vj')>=parallel_threshold && lines3D_cluster(i,7)==lines3D_cluster(j,7) )
                parallel_idx=[parallel_idx;j];
            end
        end
        lines3D_cluster(parallel_idx,:)=[];
    end
    %
    output_name =datasetname+ "/lines3D.mat";
    save(output_name,"lines3D","lines3D_cluster","cluster_degree");
end

