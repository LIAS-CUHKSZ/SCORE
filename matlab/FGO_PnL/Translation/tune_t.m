%%%%
% for each candidate translation:
% fine tune it by minimizing the square loss of inliers
% return the one with minimum loss after fine tuning. 

%%% Inputs
% t_candidates: 3XN, candidate translations
% per_rot_n_2D: Lx3, nomral vectors for 2D lines after rotation and perturbation
% p_3D        : Lx3, points on 3D lines
% epsilon_r   : scalar, error tolerance.

%%% Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
%%% License: MIT
%%%%

function t_best = tune_t(t_candidates,pert_rot_n_2D,p_3D,epsilon_t)
fine_tuned_residual_norm = inf;
t_best= zeros(3,1);
for n = 1:size(t_candidates,2)
    % find inliers under current candidate
    t_raw = t_candidates(:,n);
    residuals = sum(pert_rot_n_2D.*(p_3D-t_raw'),2);
    inliers = find(abs(residuals)<epsilon_t);

    % solve a least squares problem
    A = pert_rot_n_2D(inliers,:);
    b = sum(A.*p_3D(inliers,:),2);
    t_fine_tuned = pinv(A'*A)*(A'*b); 
    temp_ = norm(A*t_fine_tuned-b);
    
    % update the best translation
    if fine_tuned_residual_norm>temp_
        fine_tuned_residual_norm=temp_;
        t_best = t_fine_tuned;
    end
end
end

