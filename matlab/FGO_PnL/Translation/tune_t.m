function t_best = tune_t(t_candidates,pert_rot_n_2D,p_3D,epsilon_t)
fine_tuned_residual_norm = inf;
t_best= zeros(3,1);
for n = 1:size(t_candidates,2)
    t_raw = t_candidates(:,n);
    residuals = sum(pert_rot_n_2D.*(p_3D-t_raw'),2);
    inliers = find(abs(residuals)<epsilon_t);
    A = pert_rot_n_2D(inliers,:);
    b = sum(A.*p_3D(inliers,:),2);
    t_fine_tuned = pinv(A'*A)*(A'*b);
    temp_ = norm(A*t_fine_tuned-b);
    if fine_tuned_residual_norm>temp_
        fine_tuned_residual_norm=temp_;
        t_best = t_fine_tuned;
    end
end
end

