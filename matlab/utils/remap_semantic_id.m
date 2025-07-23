function [lines2D,lines3D_sub]=remap_semantic_id(lines2D,lines3D_sub,remapping)
if isempty(remapping)
    return
else
    for i = 1:size(remapping,1)
        remapped_idx = lines2D(:,4)==remapping(i,1);
        lines2D(remapped_idx,4)=remapping(i,2);
        remapped_idx = lines3D_sub(:,7)==remapping(i,1);
        lines3D_sub(remapped_idx,7)=remapping(i,2);
    end
end
end

