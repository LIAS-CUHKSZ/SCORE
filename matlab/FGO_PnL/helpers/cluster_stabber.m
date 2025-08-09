%%%%
% cluster the stabbers according to proximity

%%% Inputs
% stabbers: N x 1 (sorted)
% thres   : scalar

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%


function stabber_clustered=cluster_stabber(stabbers,thres)
% if the stabber is unique, return.
if isscalar(stabbers)
    stabber_clustered=stabbers;
    return
end

% initialize the buffer to store the stabbers under investigation
stabber_buffer = zeros(1,length(stabbers));
stabber_buffer(1) = stabbers(1);
stabber_count = 1; % count the number of stabbers in the buffer

% intialize the buffer to store cluster centers
stabber_clustered=zeros(1,length(stabbers)); 
cluster_count = 0; % count the number of clusters

% go through all stabbers
% we assume the stabbers are sorted (as the output of interval stabbing) 
for n=2:length(stabbers)
    new_stabber=stabbers(n);
    if new_stabber-stabber_buffer(1)>thres 
        % case 1: the difference with the current stabber head is too large
        % get median stabber in the buffer
        temp_idx = stabber_count-1+mod(stabber_count,2); % return an odd number
        median_stabber = median(stabber_buffer(1:temp_idx));
        % push a new cluster into the cluster buffer
        cluster_count = cluster_count+1; % number of clusters ++
        stabber_clustered(cluster_count)=median_stabber;

        % clear the stabber buffer
        stabber_buffer(1:stabber_count) = 0;
        stabber_count = 1;
    else
        % case 2: the difference with the current stabber head is within threshold
        stabber_count = stabber_count+1;
    end
    % push in the current stabber
    stabber_buffer(stabber_count) = new_stabber; % start a new cluster
end
    temp_idx = 1:(stabber_count-1+mod(stabber_count,2));
    cluster_count = cluster_count+1;
    stabber_clustered(cluster_count)=median(stabber_buffer(temp_idx)); % record the median of current cluster
    stabber_clustered(cluster_count+1:end)=[];
end