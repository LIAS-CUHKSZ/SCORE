function [far,near] = interval_projection(a, interval)
%INTERVAL_PROJECTION 此处显示有关此函数的摘要
%   a, and x in [b,c];
%  return  far= max |x-a| , near= min|x-a| 
    if a <interval(1)
        far = interval(2);
        near = interval(1);
    elseif a<= (interval(1)+interval(2))/2
        far = interval(2);
        near= a;
    elseif a<=interval(2)
        far = interval(1);
        near = a;
    else
        far = interval(1);
        near= interval(2);
    end
end

