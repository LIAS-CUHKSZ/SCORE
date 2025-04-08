function [out,R_opt] = min_error(num_candidate,R_candidates_transpose,R_gt)
    out=360;
    for c=1:num_candidate
        R_temp=R_candidates_transpose(3*(c-1)+1:3*c,:);
        R_temp=R_temp';
        error_temp=real(angular_distance(R_temp,R_gt));
        if error_temp<out
            out=error_temp;
            R_opt=R_temp;
        end
    end
end