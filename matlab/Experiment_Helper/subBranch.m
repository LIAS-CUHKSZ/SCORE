function out=subBranch(branch)
%对输入的区间进行2^2分支
%   a是下界
%   b是上界

a=branch(1:2);
b=branch(3:end);
c=0.5*(a+b);
M=[a,c,b];
out=zeros(4,4);
for i=1:4
    out(1,i)= M(1,bitget(i,1)+1);
    out(2,i)= M(2,bitget(i,2)+1);
    out(3,i)= M(1,bitget(i,1)+2);
    out(4,i)= M(2,bitget(i,2)+2);
end

end