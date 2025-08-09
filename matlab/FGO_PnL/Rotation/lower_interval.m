%%%%
%   in order to solve interval for theta satisfying 
%   |A sin(theta + phi )+ const| <=epsilon 

%%% Inputs:
% A:        Nx1
% phi:      Nx1
% const:    Nx1
% epsilon:  scalar

%%% Author: Xiang Zheng   <224045013@link.cuhk.edu.cn>
%%% License: MIT

function [interval] = lower_interval(A,phi,const,epsilon)
    interval = [];
    c_up = -const + epsilon;
    c_lo = -const - epsilon;

    if c_up<=-A 
        return;
    elseif c_up<=0
        if c_lo<=-A
            m = asin(c_up / A);
            m_l = pi - m;
            m_r = 2*pi + m;
            if phi<=-m || phi>=m_r
                return;
            else
                interval = [max(0,m_l-phi); min(pi,m_r-phi)];
            end
        else 
            m = asin(c_up / A);
            n = asin(c_lo / A);
            m_l = pi - m;
            n_l = pi - n;
            m_r = 2*pi + n;
            n_r = 2*pi + m;
            if phi<=-m || phi>=n_r
                return;
            elseif phi<=pi+n 
                interval = [m_l-phi;min(pi,n_l-phi)];
            elseif phi<=n_l
                interval = [max(m_l-phi,0);n_l-phi;m_r-phi;min(pi,n_r-phi)];
            else
                interval = [max(m_r-phi,0);min(pi,n_r-phi)];
            end
        end
    elseif  c_up<=A
        if c_lo<=-A
            m = asin(c_up/A);
            if phi <= m
                interval = [0 ; m-phi; pi-m-phi; pi];
            elseif phi<=2*pi-m 
                interval = [ max(0,pi-m-phi);min(pi,2*pi+m-phi)];

            else 
                interval = [0; 2*pi+m-phi; 3*pi-m-phi; pi];
            end
        elseif c_lo<=0
            m = asin(c_up/A);
            n = asin(c_lo/A);
            m_r = pi-m;
            n_l = pi-n;
            n_r = 2*pi +n;
            if phi<m
                interval = [0;m-phi;m_r-phi;min(pi,n_l-phi)];
            elseif phi<= n_r-pi
                interval = [max(0,m_r-phi);min(pi,n_l-phi)];
            elseif phi <=n_l
                interval = [max(0,m_r-phi);n_l-phi;n_r-phi;min(pi,2*pi+m-phi)];
            elseif phi <=m_r+pi
                interval = [max(0,n_r-phi);min(pi,2*pi+m-phi)];
            else 
                interval = [max(0,n_r-phi);2*pi+m-phi; 3*pi-m-phi;pi];
            end
        else
            m = asin(c_up/A);
            n = asin(c_lo/A);
            m_r_1= pi-m;
            n_r_1= pi-n;
            if phi<=m
                interval = [max(0,n-phi);m-phi;m_r_1-phi;n_r_1-phi];
            elseif phi<=pi+n && phi>=n_r_1
                return;
            elseif phi<=n_r_1
                interval = [max(0,m_r_1-phi);n_r_1-phi];
            elseif phi<= m_r_1+pi
                interval = [2*pi+n-phi ; min(pi,2*pi+m-phi)];
            else 
                interval = [2*pi+n-phi ; 2*pi+m-phi;m_r_1+2*pi-phi;min(pi,n_r_1+2*pi-phi)];
            end
        end
    else
        if c_lo<=-A
            interval= [0;pi];
        elseif c_lo<=0
            m = asin(c_lo/A);
            m_l = pi - m;
            m_r = 2*pi+m;
            if phi<= m_r-pi
                interval = [0;min(m_l-phi,pi)];
            elseif phi>=m_l
                interval = [max(0,m_r-phi);pi];
            else
                interval = [ 0; m_l-phi;  m_r-phi; pi];
            end

        elseif c_lo<A
            m = asin(c_lo/A);
            if phi<=pi - m
                interval = [max(0,m-phi);pi-m-phi];
            elseif phi>=pi+m
                interval = [2*pi+m - phi;min(pi,3*pi-m-phi)]; 
            else
                interval = [];
                return;
            end
        else 
            return;
        end
    end
    % test_flag = test_lower_interval(interval,A,phi,const,epsilon);
    % if ~test_flag
    %     test_flag
    % end
end


function [test_flag] = test_lower_interval(intervals,A_,phi_,const_,epsilon)
    test_flag = true;
    num = length(intervals)/2;
    for n=1:num
        inter_l = intervals(n*2-1);
        inter_r = intervals(n*2);
        test_points = linspace(inter_l,inter_r,10);
        value = A_*sin(phi_+test_points)+const_;
        if abs(value) > epsilon
            test_flag = false;
        end
    end
end

