%%%%
% Implementation of saturated interval stabbing

%%% Inputs:
% Intervals: 2Lx1, specify the endpoints for L intervals in sequence
% ids: Lx1, the group id for each interval
% kernel: the adopted saturation function

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%           Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% Version: 1.0
%%% License: MIT
%%%%

function [saturated_num_stabbed,stabber] = saturated_interval_stabbing(Intervals,ids,kernel)
    persistent kernel_buff; % only need to compute the buffer once
    if isempty(kernel_buff)
        kernel_buff=zeros(100,1);
        for i=1:100
            kernel_buff(i)=kernel(i);
        end
    end
    % Intervals: 2L * 1
    L = size(Intervals, 1) / 2;
    masks = repmat([0;1], L, 1);
    [~, sidx] = sort(Intervals);
    length_sidx = 2*L;
    sat_count = 0;
    % best_1_count = 0;
    saturated_num_stabbed = 0; 
    stabber = 0;
    count_buffer=zeros(max(ids),1);
    for i = 1:length_sidx
        if ~masks(sidx(i))
            temp = ids((sidx(i)+1)/2);
            count_buffer(temp)=count_buffer(temp)+1;
            sat_count = sat_count + kernel_buff(min(count_buffer(temp),100));
            if sat_count > saturated_num_stabbed                   
               saturated_num_stabbed = sat_count;
               % best_1_count = sum(count_buffer>0);
               stabber = Intervals(sidx(i))+1e-12;  
            end            
        else
            temp = ids(sidx(i)/2);
            sat_count = sat_count - kernel_buff(min(count_buffer(temp),100));
            count_buffer(temp)=count_buffer(temp)-1;
        end       
    end
end

