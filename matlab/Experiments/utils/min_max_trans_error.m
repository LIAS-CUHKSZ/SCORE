function [min_err,max_err,t_max,t_min] = min_max_trans_error(num_candidate,t_candidates,t_gt)
    max_err=-1; min_err=100; t_min=zeros(3,1);
    for k=1:num_candidate
        t_candi = t_candidates(:,k);
        err_ = norm(t_candi-t_gt);
        if err_>max_err
           t_max = t_candi;
           max_err = err_;
        end
        if err_<min_err
           t_min = t_candi;
           min_err = err_;
        end
        min_err = min(min_err,err_);       
    end
end

