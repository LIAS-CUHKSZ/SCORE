% fetched from github repo: https://github.com/tyhuang98/EGO_PnL
function [interval] = get_interval(A, phi, c, epsilon)

interval = [];


c_up = -c + epsilon;
c_lo = -c - epsilon;

if c_up <= -A 
    return;
elseif c_up <= 0

    if c_lo <= -A
        m = asin(c_up / A);
        m_l = pi - m;
        m_u = 2*pi + m;

        if phi <= m_l
            theta_l = m_l - phi;
            theta_u = m_u - phi;
            interval = [theta_l; theta_u];
        elseif phi <= m_u
            theta_l_1 = 0;
            theta_u_1 = m_u - phi;

            theta_l_2 = m_l + 2*pi - phi;
            theta_u_2 = 2*pi;
            
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        else
            theta_l = m_l + 2*pi - phi;
            theta_u = m_u + 2*pi - phi;
            interval = [theta_l; theta_u];
        end

    else
        m = asin(c_up / A);
        n = asin(c_lo / A);

        m_1 = pi - m;
        n_1 = pi - n;

        m_2 = 2*pi + n;
        n_2 = 2*pi + m;

        if phi <= m_1
            theta_l_1 = m_1 - phi;
            theta_u_1 = n_1 - phi;

            theta_l_2 = m_2 - phi;
            theta_u_2 = n_2 - phi;
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        elseif phi <= n_1
            theta_l_1 = 0;
            theta_u_1 = n_1 - phi;

            theta_l_2 = m_2 - phi;
            theta_u_2 = n_2 - phi;

            theta_l_3 = m_1 + 2*pi -phi;
            theta_u_3 = 2*pi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2; theta_l_3; theta_u_3];
        elseif phi <= m_2
            theta_l_1 = m_2 - phi;
            theta_u_1 = n_2 - phi;

            theta_l_2 = m_1 + 2*pi - phi;
            theta_u_2 = n_1 + 2*pi - phi;
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        elseif phi <= n_2
            theta_l_1 = 0;
            theta_u_1 = n_2 - phi;

            theta_l_2 = m_1 + 2*pi - phi;
            theta_u_2 = n_1 + 2*pi - phi;

            theta_l_3 = m_2 + 2*pi - phi;
            theta_u_3 = 2*pi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2; theta_l_3; theta_u_3];
        else
            theta_l_1 = m_1 + 2*pi - phi;
            theta_u_1 = n_1 + 2*pi - phi;

            theta_l_2 = m_2 + 2*pi - phi;
            theta_u_2 = n_2 + 2*pi - phi;
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        end

    end

elseif c_up <= A

    if c_lo <= -A

        can_l = asin(c_up / A);
        can_u = pi - can_l;

        if phi <= can_l
            theta_l_1 = 0;
            theta_u_1 = can_l - phi;

            theta_l_2 = can_u - phi;
            theta_u_2 = 2*pi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        elseif phi <= can_u
            theta_l = can_u - phi;
            theta_u = 2*pi + can_l - phi;
            
            interval = [theta_l; theta_u];
        else
            theta_l_1 = 0;
            theta_u_1 = 2*pi + can_l - phi;

            theta_l_2 = 2*pi + can_u - phi;
            theta_u_2 = 2*pi;
            
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        end

    elseif c_lo <= 0

        can_u_1 = asin(c_up / A);
        can_u_2 = pi - can_u_1;

        can_l_1 = pi - asin(c_lo / A);
        can_l_2 = 3*pi - can_l_1;

        if phi <= can_u_1
            theta_l_1 = 0;
            theta_u_1 = can_u_1 - phi;

            theta_l_2 = can_u_2 - phi;
            theta_u_2 = can_l_1 - phi;

            theta_l_3 = can_l_2 - phi;
            theta_u_3 = 2*pi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2; theta_l_3; theta_u_3];
        elseif phi <= can_u_2
            theta_l_1 = can_u_2 - phi;
            theta_u_1 = can_l_1 - phi;

            theta_l_2 = can_l_2 - phi;
            theta_u_2 = 2*pi + can_u_1 - phi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        elseif phi <= can_l_1
            theta_l_1 = 0;
            theta_u_1 = can_l_1 - phi;

            theta_l_2 = can_l_2 - phi;
            theta_u_2 = 2* pi + can_u_1 - phi;

            theta_l_3 = 2*pi + can_u_2 - phi;
            theta_u_3 = 2*pi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2; theta_l_3; theta_u_3];

        elseif phi <= can_l_2
            theta_l_1 = can_l_2 - phi;
            theta_u_1 = can_u_1 + 2*pi - phi;

            theta_l_2 = can_u_2 + 2*pi - phi;
            theta_u_2 = can_l_1 + 2*pi - phi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];

        else 
            theta_l_1 = 0;
            theta_u_1 = can_u_1 + 2*pi - phi;

            theta_l_2 = can_u_2 + 2*pi - phi;
            theta_u_2 = can_l_1 + 2*pi - phi;

            theta_l_3 = can_l_2 + 2*pi - phi;
            theta_u_3 = 2*pi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2; theta_l_3; theta_u_3];
        end

    else

        can_l_1 = asin(c_lo / A);
        can_l_2 = asin(c_up / A);

        can_u_1 = pi - can_l_2;
        can_u_2 = pi - can_l_1;

        if phi <= can_l_1
            theta_l_1 = can_l_1 - phi;
            theta_u_1 = can_l_2 - phi;

            theta_l_2 = can_u_1 - phi;
            theta_u_2 = can_u_2 - phi;
            
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];

        elseif phi <= can_l_2
            theta_l_1 = 0;
            theta_u_1 = can_l_2 - phi;

            theta_l_2 = can_u_1 - phi;
            theta_u_2 = can_u_2 - phi;

            theta_l_3 = can_l_1 + 2*pi - phi;
            theta_u_3 = 2*pi;
        
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2; theta_l_3; theta_u_3];

        elseif phi <= can_u_1

            theta_l_1 = can_u_1 - phi;
            theta_u_1 = can_u_2 - phi;
            
            theta_l_2 = 2*pi + can_l_1 - phi;
            theta_u_2 = 2*pi + can_l_2 - phi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
            
        elseif phi <= can_u_2

            theta_l_1 = 0;
            theta_u_1 = can_u_2 - phi;

            theta_l_2 = 2*pi + can_l_1 - phi;
            theta_u_2 = 2*pi + can_l_2 - phi;

            theta_l_3 = 2*pi + can_u_1 - phi;
            theta_u_3 = 2*pi;
        
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2; theta_l_3; theta_u_3];

        else

            theta_l_1 = 2*pi + can_l_1 - phi;
            theta_u_1 = 2*pi + can_l_2 - phi;

            theta_l_2 = 2*pi + can_u_1 - phi;
            theta_u_2 = 2*pi + can_u_2 - phi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];

        end

    end

else

    if c_lo <= -A
        theta_l = 0;
        theta_u = 2*pi;
        interval = [theta_l; theta_u];

    elseif c_lo <= 0

        can_l_1 = pi - asin(c_lo / A);
        can_l_2 = 3*pi - can_l_1;

        if phi <= can_l_1
            theta_l_1 = 0;
            theta_u_1 = can_l_1 - phi;
    
            theta_l_2 = can_l_2 - phi;
            theta_u_2 = 2*pi;
    
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        elseif phi <= can_l_2
            theta_l = can_l_2 - phi;
            theta_u = can_l_1 + 2*pi - phi;

            interval = [theta_l; theta_u];
        else
            theta_l_1 = 0;
            theta_u_1 = can_l_1 + 2*pi - phi;
    
            theta_l_2 = can_l_2 + 2*pi - phi;
            theta_u_2 = 2*pi;
    
            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        end

    elseif c_lo <= A
        
        can_l_1 = asin(c_lo / A);
        can_l_2 = pi - can_l_1;

        if phi <= can_l_1

            theta_l = can_l_1 - phi;
            theta_u = can_l_2 - phi;
        
            interval = [theta_l; theta_u];

        elseif phi <= can_l_2

            theta_l_1 = 0;
            theta_u_1 = can_l_2 - phi;

            theta_l_2 = can_l_1 + 2*pi - phi;
            theta_u_2 = 2*pi;

            interval = [theta_l_1; theta_u_1; theta_l_2; theta_u_2];
        else

            theta_l = 2*pi + can_l_1 - phi;
            theta_u = 2*pi + can_l_2 - phi;

            interval = [theta_l; theta_u];
        end   

    else
        return;

    end


end


end

