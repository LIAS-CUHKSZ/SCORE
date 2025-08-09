% perturb the n vectors to be orthgonal to the v vectors 
%%% Inputs:
% n:         N  x 3
% v:         N  x 3
function n_pert = pert_n(n,v)
    M = size(n,1);
    n_pert = zeros(M,3);
    for m=1:M
        ni = n(m,:);  vi = v(m,:);
        ni = (eye(3)-vi'*vi)*ni'; ni = ni'/norm(ni);
        n_pert(m,:)=ni;
    end
end