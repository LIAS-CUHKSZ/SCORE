function [c]=intersect_interval(a, b)
    if a(2) < b(1) || b(2) < a(1)
        c = [];
    else
        c = [max(a(1), b(1)); min(a(2), b(2))];
    end
end