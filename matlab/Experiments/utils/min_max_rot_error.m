function [min_err,max_err,R_min,R_max] = min_max_rot_error(num_candidate,R_candidates,R_gt)
    min_err=360; max_err=0;
    for c=1:num_candidate
        R_temp=R_candidates(3*(c-1)+1:3*c,:);
        err_temp=real(angular_distance(R_temp,R_gt));
        if err_temp<min_err
            min_err=err_temp;
            R_min=R_temp;
        end
        if err_temp>max_err
           max_err = err_temp;
           R_max = R_temp;
        end
    end
end