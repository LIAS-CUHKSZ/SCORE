%%%%
% check whether a line cross the image rectangle
% p1, p2: 2x1 endpoints pixel coordinate
% w,  h : int image width and height

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%
function isIntersect = checkLineRect(p1, p2, w, h)
isIn = @(p) (p(1)>=0 && p(1)<=w && p(2)>=0 && p(2)<=h);
if isIn(p1) || isIn(p2)
    isIntersect = true; 
    return;
end

edges = [0 0 0 h; w 0 w h; 0 0 w 0; 0 h w h];

for i = 1:4
    a = edges(i,1:2); b = edges(i,3:4);
    d1 = ccw(p1,p2,a); d2 = ccw(p1,p2,b);
    d3 = ccw(a,b,p1); d4 = ccw(a,b,p2);
    if d1*d2<0 && d3*d4<0
        isIntersect = true; return;
    end
    if (d1==0 && onSeg(p1,p2,a)) || (d2==0 && onSeg(p1,p2,b)) ...
            || (d3==0 && onSeg(a,b,p1)) || (d4==0 && onSeg(a,b,p2))
        isIntersect = true; return;
    end
end
isIntersect = false;
end

function val = ccw(a, b, c)
val = (b(1)-a(1))*(c(2)-a(2)) - (b(2)-a(2))*(c(1)-a(1));
end

function on = onSeg(a, b, c)
on = min(a(1),b(1)) <= c(1) && c(1) <= max(a(1),b(1)) && ...
    min(a(2),b(2)) <= c(2) && c(2) <= max(a(2),b(2));
end