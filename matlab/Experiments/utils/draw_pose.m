function draw_pose(T_gt,size,scale)
    % Extract the translation components from the transformation matrix
    T_translation = T_gt(1:3, 4);
    % Extract the rotation components from the transformation matrix
    T_rotation = T_gt(1:3, 1:3)*scale;
    % Plot the pose
    scatter3(T_translation(1), T_translation(2), T_translation(3), size,'ko','filled');
    hold on
    % Plot the rotation vectors
    quiver3(T_translation(1), T_translation(2), T_translation(3), T_rotation(1,1), T_rotation(2,1), T_rotation(3,1), 'r');
    hold on
    quiver3(T_translation(1), T_translation(2), T_translation(3), T_rotation(1,2), T_rotation(2,2), T_rotation(3,2), 'g');
    hold on
    quiver3(T_translation(1), T_translation(2), T_translation(3), T_rotation(1,3), T_rotation(2,3), T_rotation(3,3), 'b');
end