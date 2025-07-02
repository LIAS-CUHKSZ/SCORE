% stabbers: N x 1
% thres   : scalar
function stabber_clustered=cluster_stabber(stabbers,thres)
if isscalar(stabbers)
    stabber_clustered=stabbers;
    return
end
stabber_buffer = zeros(1,length(stabbers));
stabber_buffer(1) = stabbers(1);
valid_stabber = 1;
stabber_clustered=zeros(1,length(stabbers)); 
valid_cluster = 0;
for n=2:length(stabbers)
    new_stabber=stabbers(n);
    if new_stabber-stabber_buffer(1)>thres % the difference with the current stabber head is too large
        temp_idx = 1:(valid_stabber-1+mod(valid_stabber,2));
        valid_cluster = valid_cluster+1;
        stabber_clustered(valid_cluster)=median(stabber_buffer(temp_idx)); % record the median of current cluster
        % initialize buffer
        stabber_buffer(1:valid_stabber) = 0;
        stabber_buffer(1) = new_stabber; % start a new cluster
        valid_stabber = 1;
    else
        valid_stabber = valid_stabber+1;
        stabber_buffer(valid_stabber) = new_stabber;
    end
end
    temp_idx = 1:(valid_stabber-1+mod(valid_stabber,2));
    valid_cluster = valid_cluster+1;
    stabber_clustered(valid_cluster)=median(stabber_buffer(temp_idx)); % record the median of current cluster
    stabber_clustered(valid_cluster+1:end)=[];
end