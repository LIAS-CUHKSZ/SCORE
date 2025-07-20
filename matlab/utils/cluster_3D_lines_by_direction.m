function lines3D_cluster = cluster_3D_lines_by_direction(lines3D,cluster_degree)
parallel_threshold=cosd(cluster_degree);
lines3D_cluster = lines3D;
    i=0;
    while(i<length(lines3D_cluster))
        i=i+1;
        vi=lines3D_cluster(i,4:6)-lines3D_cluster(i,1:3);
        vi = vi / norm(vi);
        parallel_idx=[];
        for j=i+1:length(lines3D_cluster)
            vj=lines3D_cluster(j,4:6)-lines3D_cluster(j,1:3);
            vj = vj/norm(vj);
            if ( abs(vi*vj')>=parallel_threshold && lines3D_cluster(i,7)==lines3D_cluster(j,7) )
                parallel_idx=[parallel_idx;j];
            end
        end
        lines3D_cluster(parallel_idx,:)=[];
    end
end

