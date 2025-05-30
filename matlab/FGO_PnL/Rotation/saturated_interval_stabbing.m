%%%%
% Implementation of saturated interval stabbing

%%% Inputs:
% Intervals: 2Lx1, specify the endpoints for L intervals in sequence
% ids: Lx1, the group id for each interval
% kernel_buff: store weights given by the selected saturation function.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [saturated_num_stabbed,stabber] = saturated_interval_stabbing(Intervals,ids,kernel_buff)
    trunc_num=length(kernel_buff);
    % Intervals: 2L * 1
    L = size(Intervals, 1) / 2;
    masks = repmat([0;1], L, 1);
    [~, sidx] = sort(Intervals);
    length_sidx = 2*L;
    sat_count = 0;
    saturated_num_stabbed = 0; 
    stabber = [];
    count_buffer=zeros(max(ids),1);
    for i = 1:length_sidx-1
        if ~masks(sidx(i)) % start of an interval
            temp = ids((sidx(i)+1)/2);
            count_buffer(temp)=count_buffer(temp)+1;
            if count_buffer(temp)<=trunc_num
                sat_count = sat_count + kernel_buff(count_buffer(temp));
            end
            if sat_count > saturated_num_stabbed                   
               saturated_num_stabbed = sat_count;
               stabber = [Intervals(sidx(i))+1e-12,Intervals(sidx(i+1))-1e-12];
            elseif sat_count == saturated_num_stabbed
               stabber = [stabber,Intervals(sidx(i))+1e-12,Intervals(sidx(i+1))-1e-12]; 
            end            
        else % end of an interval
            temp = ids(sidx(i)/2);
            if count_buffer(temp)<=trunc_num
                sat_count = sat_count - kernel_buff(count_buffer(temp));
            end
            count_buffer(temp)=count_buffer(temp)-1;
        end       
    end
end

