%% generate boundary
sample_reso=0.01;
branch=[0;0;1;1];
if branch(3)-branch(1)>=sample_reso
    y_grid = branch(1):sample_reso:branch(3);
    z_grid = branch(2):sample_reso:branch(4);
    if y_grid(end)~=branch(3)
       y_grid=[y_grid,branch(3)];
       z_grid=[z_grid,branch(4)];
    end
else
    y_grid = [branch(1),branch(3)];
    z_grid = [branch(2),branch(4)];
end
M = length(y_grid); N = length(z_grid);
boundary = zeros(2,(M+N)*2-4);
boundary(:,1:N)=[y_grid(1)*ones(1,N);z_grid];
boundary(:,N:N+M-1)=[y_grid;z_grid(end)*ones(1,M)];
boundary(:,N+M-1:2*N+M-2)=[y_grid(end)*ones(1,N);z_grid(end:-1:1)];
boundary(:,2*N+M-2:(M+N)*2-4)=[y_grid(end:-1:2);z_grid(1)*ones(1,M-1)];
%%
n_2D_rot = [cosd(30),sind(30),0];
p_3D = [3,4,5];
epsilon_t = 0.12;
x_limit = 5;
trans_upper_interval(n_2D_rot,p_3D,epsilon_t,x_limit,boundary)