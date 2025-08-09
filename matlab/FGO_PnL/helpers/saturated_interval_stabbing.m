%%%%
% Implementation of saturated interval stabbing

%%% Inputs:
% Intervals:    2Lx1, specify the endpoints for L intervals in sequence
% ids:          Lx1,  the group id for each interval
% sat_buff:     MxN,  store weights given by the selected saturation function.
% prox_thres:   scalar, used for clustering proximate stabbers

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function [best_score,stabber] = saturated_interval_stabbing(Intervals,ids,sat_buff,prox_thres)
L = size(ids,1);
% mask = 0: the left endpoint of an interval (encoutered when entering)
% mask = 1: the right endpoint of an interval (encoutered when exiting)
masks = repmat([0;1], L, 1);
% sort all endpoints
[~, sidx] = sort(Intervals);

% initialize the buffer to store optimal stabbers
stabber = zeros(1,100000); count_valid_stabber = 0; 

% initialize buffer counting stabbed intervals for each 2D lines
count_buffer=zeros(max(ids),1);

% -------------------------------------
% --- START ---
length_sidx = 2*L;
score = 0; best_score = 0;
for i = 1:length_sidx-1

    % entering an interval
    if ~masks(sidx(i))
        temp = ids((sidx(i)+1)/2); % get 2D line id
        count_buffer(temp)=count_buffer(temp)+1; % # stabbed interval for this 2D line ++
        score = score + sat_buff(temp,count_buffer(temp)); % update score
        % update the best stabber
        if score >= best_score
            new_stabber = [Intervals(sidx(i)):prox_thres:Intervals(sidx(i+1)),Intervals(sidx(i+1))];
            num_new = length(new_stabber);
            if score > best_score
                stabber(1:num_new) = new_stabber;
                % instead of clearing the buffer
                % I record the number of optimal stabbers
                count_valid_stabber = num_new;
            else
                stabber(count_valid_stabber+1:count_valid_stabber+num_new) = new_stabber;
                count_valid_stabber = count_valid_stabber + num_new;
            end
            best_score = score;
        end

        % exiting an interval
    else
        temp = ids(sidx(i)/2);
        score = score - sat_buff(temp,count_buffer(temp));
        count_buffer(temp)=count_buffer(temp)-1;
    end
end
stabber(count_valid_stabber+1:end)=[];
end

