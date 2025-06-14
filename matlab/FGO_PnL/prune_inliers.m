function real_inliers = prune_inliers(R_,intrinsic,inliers,endpoints_3D,t_)
        %%% we filter 3D lines behind the camera and outside image under (R_,t_)
        delete = [];
        for k=1:length(inliers)
            end_point_1 = R_'*(endpoints_3D(inliers(k)*2-1,:)'-t_);
            end_point_2 = R_'*(endpoints_3D(inliers(k)*2,:)'-t_);
            if end_point_1(3) < 0 && end_point_2(3)<0 %% filter out the lines behind the camera
                delete=[delete,k];
                continue
            end
            end_point_1_pixel = intrinsic*end_point_1;
            end_point_1_pixel = end_point_1_pixel(1:2)/end_point_1_pixel(3);
            end_point_2_pixel = intrinsic*end_point_2;
            end_point_2_pixel = end_point_2_pixel(1:2)/end_point_2_pixel(3);
            intersect_flag = checkLineRect(end_point_1_pixel,end_point_2_pixel,1920,1440);
            if ~intersect_flag
                delete=[delete,k];
            end
        end
        real_inliers = inliers;
        if ~isempty(delete)
            real_inliers(delete)=[];
        end
end

