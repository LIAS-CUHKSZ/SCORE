%%% prune 3D lines behind the camera and outside image under (R_,t_)
function real_inliers = prune_inliers(R_,intrinsic,inliers,endpoints_3D,t_)
        % initialize buffer for lines to be deleted
        delete = [];
        for k=1:length(inliers)
            % prune lines behind the camera
            end_point_1 = R_'*(endpoints_3D(inliers(k)*2-1,:)'-t_);
            end_point_2 = R_'*(endpoints_3D(inliers(k)*2,:)'-t_);
            if end_point_1(3) < 0 && end_point_2(3)<0 
                delete=[delete,k];
                continue
            end

            % truncate lines behind the camera
            % end_point = w*end_point_1 + (1-w)*end_point_2
            if end_point_2(3) < 0
                w = (0.05-end_point_2(3))/(end_point_1(3)-end_point_2(3));
                end_point_2 = end_point_2 + w*(end_point_1-end_point_2);
            end

            if end_point_1(3) < 0
                w = (0.05-end_point_2(3))/(end_point_1(3)-end_point_2(3));
                end_point_1 = end_point_2 + w*(end_point_1-end_point_2);
            end

            % prune lines not intersecting the image
            end_point_1_pixel = intrinsic*end_point_1;
            end_point_1_pixel = end_point_1_pixel(1:2)/end_point_1_pixel(3);
            end_point_2_pixel = intrinsic*end_point_2;
            end_point_2_pixel = end_point_2_pixel(1:2)/end_point_2_pixel(3);
            intersect_flag = checkLineRect(end_point_1_pixel,end_point_2_pixel,1920,1440);
            if ~intersect_flag
                delete=[delete,k];
            end
        end

        % output
        real_inliers = inliers;
        if ~isempty(delete)
            real_inliers(delete)=[];
        end
end

