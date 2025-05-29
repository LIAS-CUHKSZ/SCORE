% fetched from github repo: https://github.com/tyhuang98/EGO_PnL
function [A_all, phi_all, c_all] = get_parameter(Point_M,Point_N, alpha_beta_value)

I_3 = [1, 0, 0; 0, 1, 0; 0, 0, 1];

alpha = alpha_beta_value(1);
beta = alpha_beta_value(2);

r=[sin(beta)*cos(alpha); sin(beta)*sin(alpha); cos(beta)];

skew_r = [0, -r(3), r(2); r(3), 0, -r(1);-r(2), r(1), 0];
skew_r_square = skew_r * skew_r;

N = size(Point_M,2);
a_all = zeros(N, 1);
b_all = zeros(N, 1);
c_all = zeros(N, 1);

for i = 1:N
    a_all(i) = Point_N(:, i)' * skew_r * Point_M(:, i);
    b_all(i) = -Point_N(:, i)' * skew_r_square * Point_M(:, i);
    c_all(i) = Point_N(:, i)' * (I_3 + skew_r_square) * Point_M(:, i);
end

A_all = sqrt(a_all.^2 + b_all.^2);
phi_all = atan2(b_all, a_all);
phi_all = phi_all.*(phi_all >= 0) + (phi_all+2*pi).*(phi_all < 0);

end

