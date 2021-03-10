function [I_grid] = generate_interpolation_grid_2D(N,x_inp,y_inp,order)

% % Inputs should be in ratio of the length of the element ! 

if strcmp(order,'linear')


l_inp = floor(x_inp * N);
m_inp = floor(y_inp * N);

alpha_x_inp = x_inp * N - l_inp;
alpha_y_inp = y_inp * N - m_inp;

I_grid = zeros(N,N);
I_grid(l_inp,m_inp) = (1-alpha_x_inp)*(1-alpha_y_inp);
I_grid(l_inp+1,m_inp) = alpha_x_inp*(1-alpha_y_inp);
I_grid(l_inp,m_inp+1) = (1-alpha_x_inp)*alpha_y_inp;
I_grid(l_inp+1,m_inp+1) = alpha_x_inp*alpha_y_inp;

elseif strcmp(order,'cubic')
    
    % % TO DO LOL
    
end

end