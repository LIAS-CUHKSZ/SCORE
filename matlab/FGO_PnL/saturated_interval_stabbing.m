%%%%
% Implementation of saturated interval stabbing

%%% Inputs:
% Intervals:    2Lx1, specify the endpoints for L intervals in sequence
% ids:          Lx1,  the group id for each interval
% kernel_buff:  MxN,  store weights given by the selected saturation function.
% prox_thres:   scalar, used for clustering proximate stabbers

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% Version: 2.0
%%% License: MIT
%%%%

function [best_score,stabber] = saturated_interval_stabbing(Intervals,ids,kernel_buff,prox_thres)
    L = size(Intervals, 1) / 2; 
    masks = repmat([0;1], L, 1);
    [~, sidx] = sort(Intervals);
    length_sidx = 2*L;
    score = 0; best_score = 0; 
    stabber = zeros(1,10000); valid_stabber = 0;
    count_buffer=zeros(max(ids),1);
    for i = 1:length_sidx-1
        if ~masks(sidx(i)) % start of an interval
            temp = ids((sidx(i)+1)/2);
            count_buffer(temp)=count_buffer(temp)+1;
            score = score + kernel_buff(temp,count_buffer(temp));
            if score >= best_score
               new_stabber = [Intervals(sidx(i)):prox_thres:Intervals(sidx(i+1)),Intervals(sidx(i+1))];
               num_new = length(new_stabber);
               if score > best_score
                   stabber(1:num_new) = new_stabber;
                   valid_stabber = num_new;
               else
                   stabber(valid_stabber+1:valid_stabber+num_new) = new_stabber;
                   valid_stabber = valid_stabber + num_new;
               end            
               best_score = score;
            end
        else % end of an interval
            temp = ids(sidx(i)/2);
            score = score - kernel_buff(temp,count_buffer(temp));
            count_buffer(temp)=count_buffer(temp)-1;
        end       
    end
    stabber(valid_stabber+1:end)=[];
end

