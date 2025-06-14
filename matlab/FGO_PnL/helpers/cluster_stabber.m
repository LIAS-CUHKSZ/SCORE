% stabbers: N x 1
% thres   : scalar
function stabber_clustered=cluster_stabber(stabbers,thres)
if isscalar(stabbers)
    stabber_clustered=stabbers;
    return
end
stabber_clustered=[];
N = length(stabbers);
stabber_buffer=stabbers(1);
for n=2:N
    new_stabber=stabbers(n);
    if new_stabber-stabber_buffer(1)>thres
        temp_idx = 1:(length(stabber_buffer)-1+mod(length(stabber_buffer),2));
        stabber_clustered=[stabber_clustered,median(stabber_buffer(temp_idx))];
        stabber_buffer=new_stabber;
    else
        stabber_buffer=[stabber_buffer,new_stabber];
    end
end
    temp_idx = 1:(length(stabber_buffer)-1+mod(length(stabber_buffer),2));
    stabber_clustered=[stabber_clustered,median(stabber_buffer(temp_idx))];
end