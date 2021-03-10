clc;
clear all;
close all;

% Loading some notes and freq data
[num,txt,raw] = xlsread('notes_and_freq_wavelengths.xlsx') ;
txt(1:24,:) = [];
iFreq = 22; % going for note A3, f = 220 in this example..

% % FLAGS
bowed  = 1;
connection = 1;
excitePlate = 0;
ramped = 0;
plotFlag = 0;
dampingString = 0;
dampingPlate = 1;
flagFullPlate = 0;
bowingModel = 'ElastoPlastic'; 
BC_string = 'SimplySupported';
plotFigures = 1;


% General Params
dur = 2;
Fs = 44100; %% numerical scheme does not work at fs = 44100
k = 1/Fs;

NF = floor(dur*Fs);
tVec = [0:1:NF-1].*k;

pathSave_root = cd;
pathSaveFig = fullfile(pathSave_root, '.\figures');
pathSave = fullfile(pathSave_root, '.\sounds');


% % BOWING PARAMETERS STATIC MODEL:

if strcmp(bowingModel,'Static')
    FB = 0.5; % dimensional case (it will be scaled with rho/A)
    vB = 0.1; % bow velocity (m/s)
    sig = 80; % friction law free parameter (1/m^2) (a) % decreasing sig increases the stick time 
    tol = 1e-7; % tolerance for Newton-Raphson method
    %%%%%% end global parameters
    % derived parameters
    A_NR = exp(1/2)*sqrt(2*sig);
elseif strcmp(bowingModel,'ElastoPlastic')
    % BOWING PARAMETERS ELASTO-PLASTIC:
    mu_C = 0.3;
    mu_S = 0.8;
    FB = 0.5;
    fC = FB*mu_C;
    fS = FB*mu_S;
    vB = 0.1;
    vS = 0.1;
    s0 = 1e4;
    s1 = 0.001*sqrt(s0);
    s2 = 0.4;
    s3_base = 0.00; % you multiply with FB later..
    z_ba = 0.7*(mu_C*FB)/s0;
    w_rnd_vec = -1 + 2 .* rand(NF,1);    
    tol = 1e-7; % tolerance for Newton-Raphson method
end


% % STIFF STRING:
% String location
string_start_coord = [0.0711111111111111 - 0.15  0.33];
string_end_coord = [0.568888888888889 - 0.15 0.33] ;  

x_inp_bridge_loc = 0.382; % absolute value

% % BOWING POSITION
inp_bow_pos_x = 0.29; % in percentage of L
% inp_bow_pos_x = 0.45;
% inp_bow_pos_x = 7*h / L;
% inp_bow_pos_x = 0.55;

% Dimensional params:
rho = 7850; % [kg/m^3]
r = 5e-4;
A = pi*(r)^2; % [m^2]
L = string_end_coord(1)- string_start_coord(1);
% L = 1;
E = 2e11;

I = pi*r^4/4;
K = sqrt(E*I/(rho*A)); % this is the dimensionless case..

% Tuning
f0 = str2double(txt(iFreq,2))

waveLength = 2*L;
c = f0*waveLength;

if dampingString
    sig0 = 0.1;
    sig1 = 0.005;
else
    sig0 = 0.0;
    sig1 = 0.0;
end

h = sqrt((c^2 * k^2 + 4 * sig1 * k + sqrt((c^2 * k^2 + 4 * sig1 * k)^2 + 16 * K^2 * k^2)) / 2);
N = floor(L/h)
% N = min(floor(L/h),30); % If you want to limit the number of points
h = L/N;


% % PLATE
% % DIMENSIONAL PARAMS

L_plate = 0.526; % Calibrated to match the size of the amoeba violin plate   
% % PARAMS FROM LITERATURE PAPER:
% rho_plate = 375; % Density of violin wood - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002554
rho_plate = 400; % kg/m^3 % from another paper
% H_plate = 0.007; % [m] 7 [mm] : check: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002554
H_plate = 0.001; % 
% E_plate = 11400*10^6; % [Pa] Red spruce - 12% moisture: https://amesweb.info/Materials/Youngs-Modulus-of-Wood.aspx 
E_plate = 10*10^9; % [Pa] [N/m^2]
nu_plate = 0.3; % Poisson's ratio [-] whatevs..


% % % % % PARAMS FROM TROMBA PAPER
% % % % rho_plate = 50; 
% % % % % rho_plate = 250; 
% % % % % rho_plate = 1250;
% % % % H_plate = 0.01;
% % % % % H_plate = 0.5; 
% % % % E_plate = 2*10^5;
% % % % % E_plate = 2*10^7; 
% % % % nu_plate = 0.3; % Poisson's ratio [-] whatevs..

% Derived params:
D_plate = E_plate*H_plate^3/(12*(1-nu_plate^2));
K_plate = sqrt(D_plate/(rho_plate*H_plate)); 

if dampingPlate
    sig0_plate = 0.5; % I think these are also from the tromba paper..
    sig1_plate = 0.05;
else
    sig0_plate = 0.0;
    sig1_plate = 0.0;
end



h_plate = 2*sqrt(k*(sig1_plate + sqrt(K_plate^2 + sig1_plate^2))); % Dimensional case !

if ~flagFullPlate
%     N_plate = min(floor(L_plate/h_plate),21); % Aim to match what you have in JUCE
%     N_plate = min(floor(L_plate/h_plate),30); % Aim to match what you have in JUCE
    N_plate = min(floor(L_plate/h_plate),35); % Aim to match what you have in JUCE
%     N_plate = min(floor(L_plate/h_plate),40); % Aim to match what you have in JUCE
else
    N_plate = floor(L_plate/h_plate);
end
h_plate = L_plate/N_plate;

N_plate
% generateAmoebaGrid(L_plate,N_plate);
generateAmoebaGrid_rev2(L_plate,N_plate);


% Select points for analysis (amoeba grid)
xVec = [0:N_plate-1];
yVec = [0:N_plate-1];

[X,Y] = ndgrid(xVec,yVec); % This is the correct way ! 

ana = load(['locsAll_violin_size_',num2str(N_plate),'_x_',num2str(N_plate),'.mat']);
locsDo = ana.locsDo;

figure(3);
p1 = plot(X(:).*h_plate,Y(:).*h_plate,'x');
grid on
hold all
axis equal
p2 = plot(X(locsDo).*h_plate,Y(locsDo).*h_plate,'o');
p3 = plot([string_start_coord(1) string_end_coord(1)],... % maybe make it discretized ! 
     [string_start_coord(2) string_end_coord(2)],'k-','linewidth',2);



% % DEFINE FORCE AND BOW VELOCITY ENVELOPE IN TIME

durAttack = 0.0*dur;
durSustain = 0.5*dur;
durRelease = 0.125*dur;

linear_asr = generate_linear_asr_envelope_vector(durAttack,durSustain,durRelease,dur,Fs);

% Assign envelope
FB_vec = FB.*linear_asr;
vB_vec = vB.*linear_asr;

% the exponential envelope is just applied to the output for a nicer
% sound.. I did this so the sounds in the app the girls made were better. I
% guess it was a bit of a cheat.
durAttack = 0.125*dur;
durSustain = 0.375*dur;
durRelease = 0.5*dur;
exp_curve_comb = generate_exponential_asr_envelope_vector(durAttack,durSustain,durRelease,dur,Fs);

% Plot the envelopes
figure(1);
subplot(3,1,1)
plot(tVec,FB_vec,'linewidth',2)
grid on
ylabel('fN [N]')
subplot(3,1,2)
plot(tVec,vB_vec,'linewidth',2)
grid on
xlabel('Sample no. [-]')
ylabel('vB [m/s]')
subplot(3,1,3)
plot(tVec,exp_curve_comb,'linewidth',2)
grid on
xlabel('Sample no. [-]')
ylabel('Exp Envelope [-]')


% % DISTRIBUTIONS FOR CONNECTION - PLATE

x_inp = x_inp_bridge_loc;
y_inp = string_start_coord(2); 

figure(3);
hold all
p4 = plot(x_inp,y_inp,'md','linewidth',3);
xlabel('X dir [m]')
ylabel('Y dir [m]')

legend([p1,p2,p3,p4],{'Squared 2-D grid for plate',...
                      'Violino Arpa',...
                      'Stiff String',...
                      'Bridge Location'})

% go from real dim to ratio of L -> necessary for generate_interpolation_grid_2D function                  
x_inp_ratio = x_inp / L_plate; 
y_inp_ratio = y_inp / L_plate;

% Distribution function
[I_P] = generate_interpolation_grid_2D(N_plate,x_inp_ratio,y_inp_ratio,'linear'); % ratioed over the length..

% Spreading function
J_P = I_P.*(1./h_plate^2);


% % DISTRIBUTIONS FOR CONNECTION - STRING

% again conn_point is in percentage of L
conn_point = (x_inp-string_start_coord(1))/L;

% Distribution function
[I_S] = generate_interpolation_grid_1D(N,conn_point,'cubic');

% Spreading function
J_S = 1/h * I_S;


% % PLATE OUTPUT
x_out = 0.4;
y_out = 0.15;
x_out_ratio = x_out / L_plate;
y_out_ratio = y_out / L_plate;

[I_plate_out] = generate_interpolation_grid_2D(N_plate,x_out_ratio,y_out_ratio,'linear'); % ratioed over the length..

% % STRING OUTPUT:
x_out_string = 0.45; % in ratio of L
% Distribution function
[I_grid_out] = generate_interpolation_grid_1D(N,x_out_string,'cubic');
[I_grid_2_3rd_str] = generate_interpolation_grid_1D(N,2/3,'cubic');
[I_grid_1_4th_str] = generate_interpolation_grid_1D(N,1/4,'cubic');


% Add bowing position and possible output position on plate. I think I'm
% actually outputing the sound from the bridge, i.e. connection of string
% to plate.
figure(3);
hold all
plot(x_out,y_out ,'gd','linewidth',3)

bp = inp_bow_pos_x;

plot(string_start_coord(1)+bp * L,string_start_coord(2),'rd','linewidth',3)
xlabel('Xdir [m]')
ylabel('Ydir [m]')




% % SETUP RUN

% % String
uPrev = zeros(N,1);
u = zeros(N,1); % same as uPrev but with some velocity ! 
uNext = zeros(N,1);

% In case you don't bow..
if ~bowed
    excitation = hann(4);

    uPrev(floor(N/2)-2:floor(N/2)+1) = excitation;
    u(floor(N/2)-2:floor(N/2)+1) = excitation;
end

% % Plate
wPrev = zeros(N_plate,N_plate);
w = zeros(N_plate,N_plate); % same as uPrev but with some velocity ! 
wNext = zeros(N_plate,N_plate);

% Something for possible testing.
if excitePlate
    w = I_P.*1e-3;
    wPrev = I_P.*1e-3;
end

% % Preallocation for Newton Raphson
if strcmp(bowingModel,'Static')
    vrel_last = 0;
elseif strcmp(bowingModel,'ElastoPlastic')
    fC = mu_C*FB;
    fS = mu_S*FB;
    z_prev = 0;

    r_last = vB;
    r_prev = vB;
    a_last = r_last;
    a_prev = r_prev;

    vrel_last = -vB;
    z_last = 0; 

    F_last = 0;
end

% % START FDS 
vr = zeros(NF,1);
Fvec = zeros(NF,1);
z_vec = zeros(NF,1);
Fbow_vector = zeros(NF,1);
F_vec = zeros(NF,1); 

out_at_bridge = zeros(NF,1);
out_plate = zeros(NF,1);
out_string = zeros(NF,1);
out_string_2_3rd = zeros(NF,1);
out_string_1_4th = zeros(NF,1);
for t=1:NF
    % Printing time step every now and then..
    if mod(t,10000) == 0
        t
    end

    % If I'm bowing.
    if bowed
        if ramped
            if strcmp(bowingModel,'Static')
                FB = FB_vec(t);
                if FB == 0
                    FB = 1e-7;
                end
                vB = vB_vec(t);
            elseif strcmp(bowingModel,'ElastoPlastic')
                FB = FB_vec(t);
                if FB == 0
                    FB = 0.01;
                end
                vB = vB_vec(t);

                fS = mu_S*FB;
                fC = mu_C*FB;
                z_ba = 0.7*(mu_C*FB)/s0;

                s3 = s3_base*FB;
            end
        end
        
        bp = inp_bow_pos_x;
        
        % Interpolation and spreading function for bowing position
        [I_B] = generate_interpolation_grid_1D(N,bp,'cubic'); % ratioed over the length..
        J_B = 1/h * I_B;

        % These terms are basically various interpolation distributions applied
        % to various terms in the equations. 
        % Important identity: <f,J(x0)> = I(x0)f -> inner product of some
        % function with a spreading operator is the interpolation function applied to that function
        % J = I * (1/h^dim) with dim being the dimension of the element -> 2 for plate, 1 for string
        I_B_J_B = sum(J_B(:).*J_B(:).* h); % bowing distribution applied to bowing spreading function
        I_B_J_S = sum(J_B(:).*J_S(:).* h);     
        I_S_J_S = sum(J_S(:).*J_S(:).* h);     
        I_P_J_P = sum(J_P(:).*J_P(:).* h_plate^2);     
        I_S_J_B = sum(J_S(:).*J_B(:).* h);   
        I_S_u_n = sum(J_S(:).*u(:).* h);
        I_S_u_n_min_1 = sum(J_S(:).*uPrev(:).* h);
        I_P_w_n = sum(J_P(:).*w(:).* h_plate^2);
        I_P_w_n_min_1 = sum(J_P(:).*wPrev(:).* h_plate^2);

        I_S_dxx_u_n = (1/h^2)*sum(sum(I_S(3:N-2).*(u(4:N-1)-2.*u(3:N-2)+u(2:N-3))));
        I_S_dxx_u_n_min_1 = (1/h^2)*sum(sum(I_S(3:N-2).*(uPrev(4:N-1)-2.*uPrev(3:N-2)+uPrev(2:N-3))));
        I_S_dxxxx_u_n = (1/h^4)*sum(sum(I_S(3:N-2).*(u(5:N)-4.*u(4:N-1)+6.*u(3:N-2)-4.*u(2:N-3)+u(1:N-4))));
%         IS_dxxxx_u_n_min_1 = (1/h^4)*sum(sum(I_S(3:N-2).*(uPrev(5:N)-4.*uPrev(4:N-1)+6.*uPrev(3:N-2)-4.*uPrev(2:N-3)+uPrev(1:N-4))));

        IP_delta_laplace_w_n = (1/h_plate^2)*sum(sum(I_P(:,3:N_plate-2).*(w(:,4:N_plate-1)-2.*w(:,3:N_plate-2)+w(:,2:N_plate-3))))...
                             + (1/h_plate^2)*sum(sum(I_P(3:N_plate-2,:).*(w(4:N_plate-1,:)-2.*w(3:N_plate-2,:)+w(2:N_plate-3,:))));   
        IP_delta_laplace_w_n_min_1 = (1/h_plate^2)*sum(sum(I_P(:,3:N_plate-2).*(wPrev(:,4:N_plate-1)-2.*wPrev(:,3:N_plate-2)+wPrev(:,2:N_plate-3))))...
                                   + (1/h_plate^2)*sum(sum(I_P(3:N_plate-2,:).*(wPrev(4:N_plate-1,:)-2.*wPrev(3:N_plate-2,:)+wPrev(2:N_plate-3,:))));                            
        IP_delta_laplace_x2_w_n = (1/h_plate^4)*sum(sum(I_P(:,3:N_plate-2).*(w(:,5:N_plate)-4.*w(:,4:N_plate-1)+6.*w(:,3:N_plate-2)-4.*w(:,2:N_plate-3)+w(:,1:N_plate-4))))...
                                + (1/h_plate^4)*sum(sum(I_P(3:N_plate-2,:).*(w(5:N_plate,:)-4.*w(4:N_plate-1,:)+6.*w(3:N_plate-2,:)-4.*w(2:N_plate-3,:)+w(1:N_plate-4,:))))...
                                + (2/h_plate^4)*sum(sum(I_P(3:N_plate-2,3:N_plate-2).*( w(4:N_plate-1,4:N_plate-1) - 2.*w(4:N_plate-1,3:N_plate-2) + w(4:N_plate-1,2:N_plate-3)- 2.*w(3:N_plate-2,4:N_plate-1) + 4.*w(3:N_plate-2,3:N_plate-2) - 2.*w(3:N_plate-2,2:N_plate-3) + w(2:N_plate-3,4:N_plate-1) - 2.*w(2:N_plate-3,3:N_plate-2) + w(2:N_plate-3,2:N_plate-3) )));  

        % b and q are some terms that come out from the equations.. I need
        % to write them down properlly at some point. 
        % For now see: Bowed_membrane_connected_with_acoustic_tube.pdf
        b = (k^2/(1+sig0*k))*(c^2.*I_S_dxx_u_n - K^2.*I_S_dxxxx_u_n + (2*sig1/k)*(I_S_dxx_u_n - I_S_dxx_u_n_min_1)) + ...
            (k^2/(1+sig0*k))*(2/k^2)*I_S_u_n - (k^2/(1+sig0*k))*((1-sig0*k)/k^2)*I_S_u_n_min_1 - ...
            (k^2/(1+sig0_plate*k))*(-K_plate^2.*IP_delta_laplace_x2_w_n + (2*sig1_plate/k)*(IP_delta_laplace_w_n-IP_delta_laplace_w_n_min_1)) - ...
            (k^2/(1+sig0_plate*k))*(2/k^2).*I_P_w_n + (k^2/(1+sig0_plate*k))*((1-sig0_plate*k)/k^2)*I_P_w_n_min_1;

        q = ( (-2/k)*(1/k)*((sum(u(:).*I_B(:))-sum(uPrev(:).*I_B(:)))) + ...
            (2/k)*vB +...
            2*sig0*vB -...
            (c^2)*(1/h^2)*(sum((u(4:N-1)-2.*u(3:N-2)+u(2:N-3)).*I_B(3:N-2))) +...
            K^2*(1/h^4)*(sum((u(5:N) - 4.*u(4:N-1) + 6.*u(3:N-2) - 4.*u(2:N-3) + u(1:N-4)).*I_B(3:N-2)))-...
            (2*sig1)*(1/h^2)*(1/k)*(sum((u(4:N-1)-2.*u(3:N-2)+u(2:N-3)).*I_B(3:N-2))) + ...
            (2*sig1)*(1/h^2)*(1/k)*(sum((uPrev(4:N-1)-2.*uPrev(3:N-2)+uPrev(2:N-3)).*I_B(3:N-2))) ); 


        if strcmp(bowingModel,'Static')
            % Newton-Raphson method to determine relative velocity
            eps = 1;
            vrel_last = 0;
            count = 0;
            while eps>tol && count < 100
                % to do: rewrite this so you get rid of the division by FB ! 
                vrel = vrel_last -...
                        (2/(k*FB*I_B_J_B) * vrel_last + A_NR*vrel_last*exp(-sig*vrel_last^2) + q/(FB*I_B_J_B))/...
                        (2/(k*FB*I_B_J_B) + A_NR*exp(-sig*vrel_last^2)*(1+2*sig*vrel_last^2));   
                eps = abs(vrel-vrel_last);
                vrel_last = vrel;
                count = count + 1;
            end          
            Fbow_vector(t) = FB*A_NR*vrel*exp(-sig*vrel^2);

        elseif strcmp(bowingModel,'ElastoPlastic')
            % this follows the implementation from Silvin's Elasto-Plastic
            % paper.
            % term_d_val -> means the derivative of some "term" with
            % respect to "val"
            s3 = s3_base*FB;            
            eps = 1;
            theta_last = [vrel_last;z_last]; 
            count = 0;
            w_rnd_last = -1 + (2) .* rand(1);    
            while eps>tol && count < 100
                vrel_last = theta_last(1);
                z_last = theta_last(2);

                if vrel_last == 0
                    z_ss = fS/s0;            
                else            
                    z_ss = sign(vrel_last)./s0 .* (fC + (fS - fC)*exp(-(vrel_last./vS).^2));
                end      

                if sign(vrel_last) == sign(z_last)
                    if abs(z_last)<=z_ba
                        alpha_fct = 0;
                    elseif (z_ba < abs(z_last)) && (abs(z_last) < abs(z_ss))
                        alpha_fct = 0.5*(1 + sign(z_last)*sin((pi*(z_last - sign(z_last)*0.5*(abs(z_ss)+z_ba))/(abs(z_ss)-z_ba))));
                    elseif abs(z_last) >= abs(z_ss)
                        alpha_fct = 1;
                    end
                else
                    alpha_fct = 0;
                end

                r_last = vrel_last*(1-alpha_fct*z_last/z_ss);
                a_last = ((2/k)*(z_last-z_prev)-a_prev);

                g1 = I_B_J_B*((s0*z_last+s1*r_last+s2*vrel_last+s3*w_rnd_vec(t))/(rho*A)) + (2/k + 2*sig0)*vrel_last + q;        
                g2 = r_last - ((2/k)*(z_last-z_prev)-a_prev);

                if sign(vrel_last)>= 0
                    z_ss_d_vrel = -2*vrel_last*(-fC + fS)*exp(-vrel_last^2/vS^2)/(s0*vS^2);
                else
                    z_ss_d_vrel = 2*vrel_last*(-fC + fS)*exp(-vrel_last^2/vS^2)/(s0*vS^2);
                end

                if (z_ba < abs(z_last)) && (abs(z_last) < abs(z_ss))
                    if sign(z_last)>=0
                        alpha_fct_d_vrel = 0.5*(-0.5*pi*(z_ss*z_ss_d_vrel)*sign(z_ss)/((-z_ba + abs(z_ss))*z_ss) - pi*(z_ss*z_ss_d_vrel)*(-0.5*z_ba + z_last - 0.5*abs(z_ss))*sign(z_ss)/((-z_ba + abs(z_ss))^2*z_ss))*cos(pi*(-0.5*z_ba + z_last - 0.5*abs(z_ss))/(-z_ba + abs(z_ss)));   
                        alpha_fct_d_z = 0.5*pi*cos(pi*(-0.5*z_ba + z_last - 0.5*abs(z_ss))/(-z_ba + abs(z_ss)))/(-z_ba + abs(z_ss));
                    else
                        alpha_fct_d_vrel = -0.5*(0.5*pi*(z_ss*z_ss_d_vrel)*sign(z_ss)/((-z_ba + abs(z_ss))*z_ss) - pi*(z_ss*z_ss_d_vrel)*(0.5*z_ba + z_last + 0.5*abs(z_ss))*sign(z_ss)/((-z_ba + abs(z_ss))^2*z_ss))*cos(pi*(0.5*z_ba + z_last + 0.5*abs(z_ss))/(-z_ba + abs(z_ss)));
                        alpha_fct_d_z = -0.5*pi*cos(pi*(0.5*z_ba + z_last + 0.5*abs(z_ss))/(-z_ba + abs(z_ss)))/(-z_ba + abs(z_ss));     
                    end
                else
                    alpha_fct_d_vrel = 0;
                    alpha_fct_d_z = 0;
                end

                r_last_d_vrel = vrel_last*(z_last*alpha_fct*z_ss_d_vrel/z_ss^2 - z_last*alpha_fct_d_vrel/z_ss) - z_last*alpha_fct/z_ss + 1;
                r_last_d_z = vrel_last*(-z_last*alpha_fct_d_z/z_ss - alpha_fct/z_ss);

                g1_d_vrel = 2*sig0 + 2/k + I_B_J_B*(s1*r_last_d_vrel + s2)/(rho*A);
                g1_d_z = I_B_J_B*(s0 + s1*r_last_d_z)/(rho*A);

                g2_d_vrel = r_last_d_vrel;
                g2_d_z = r_last_d_z -2/k;

                Jacobian_matrix = [g1_d_vrel, g1_d_z;
                                   g2_d_vrel, g2_d_z];

                theta = theta_last - Jacobian_matrix^-1*[g1;g2];

                eps = norm(theta-theta_last);

                theta_last = theta;

                count = count+1;
                if count == 99
                    disp(['count is: ',num2str(count)]);
                end        
            end

            % Once the NR solver converges use the found values of vrel and
            % z to calculate the necessary params that go in the Fbow
            % equation.
            vrel = theta(1);
            z = theta(2);

            w_rnd = w_rnd_vec(t);

            if vrel == 0
                z_ss = fS/s0;            
            else            
                z_ss = sign(vrel)./s0 .* (fC + (fS - fC)*exp(-(vrel./vS).^2));
            end      

            if sign(vrel) == sign(z)
                if abs(z)<=z_ba
                    alpha_fct = 0;
                elseif (z_ba < abs(z)) && (abs(z) < abs(z_ss))
                    alpha_fct = 0.5*(1 + sign(z)*sin((pi*(z - sign(z)*0.5*(abs(z_ss)+z_ba))/(abs(z_ss)-z_ba))));
                elseif abs(z) >= abs(z_ss)
                    alpha_fct = 1;
                end
            else
                alpha_fct = 0;
            end

            r_last = vrel*(1-alpha_fct*z/z_ss);

            r = r_last;
            a = (2/k)*(z-z_prev) - a_prev;

            Fbow_vector(t) = (s0*z+s1*r+s2*vrel+s3*w_rnd);
            
            z_prev = z;
            a_prev = a;       

            z_vec(t) = z;              
            
        end
    
        % % Connection force:
        F = (-b+(k^2/(1+sig0*k))*I_S_J_B*((Fbow_vector(t))/(rho*A)))/(k^2*I_S_J_S/((1+sig0*k)*rho*A) + k^2*I_P_J_P/((1+sig0_plate*k)*rho_plate*H_plate));
        if ~connection
            F = 0;
        end        
        
        % % Update equations -> string
        uNext(3:N-2) = (k^2/(1+sig0*k))*( (c^2/h^2)*(u(4:N-1) - 2*u(3:N-2) + u(2:N-3))...
                        +(2*sig1/(k*h^2))*(u(4:N-1) - 2*u(3:N-2) + u(2:N-3) - uPrev(4:N-1) + 2*uPrev(3:N-2) - uPrev(2:N-3))...
                        - K^2*(1/h^4)*(u(5:N) - 4*u(4:N-1) + 6*u(3:N-2) - 4*u(2:N-3) + u(1:N-4))...
                        - J_B(3:N-2).*Fbow_vector(t)./(rho*A) + J_S(3:N-2)*F/(rho*A) ...
                        + (2/k^2)*u(3:N-2) - (1-sig0*k).*uPrev(3:N-2)./k^2  );
       
        if strcmp(BC_string,'SimplySupported')
            uNext(2) = (k^2/(1+sig0*k))*( (c^2/h^2)*(u(3) - 2*u(2) + u(1))...
                            +(2*sig1/(k*h^2))*(u(3) - 2*u(2) + u(1) - uPrev(3) + 2*uPrev(2) - uPrev(1))...
                            - K^2*(1/h^4)*(u(4) - 4*u(3) + 6*u(2) - 4*u(1) - u(2))...
                            - J_B(2).*Fbow_vector(t)./(rho*A) + J_S(2)*F/(rho*A) ...
                            + (2/k^2)*u(2) - (1-sig0*k).*uPrev(2)./k^2  );                    

            uNext(N-1) = (k^2/(1+sig0*k))*( (c^2/h^2)*(u(N) - 2*u(N-1) + u(N-2))...
                            +(2*sig1/(k*h^2))*(u(N) - 2*u(N-1) + u(N-2) - uPrev(N) + 2*uPrev(N-1) - uPrev(N-2))...
                            - K^2*(1/h^4)*(-u(N-1) - 4*u(N) + 6*u(N-1) - 4*u(N-2) + u(N-3))...
                            - J_B(N-1).*Fbow_vector(t)./(rho*A) + J_S(N-1)*F/(rho*A) ...
                            + (2/k^2)*u(N-1) - (1-sig0*k).*uPrev(N-1)./k^2  );
        end
             
        % Update equations -> Plate
        for iAll = 1:length(locsDo)

            iUx = rem(locsDo(iAll)-1,size(wNext,1))+1; % rem is remainder after division. rem(23,5) = 3
            iUy = (locsDo(iAll)-iUx)/size(wNext,1) + 1;

            fac1 = (w(iUx+2,iUy)-4*w(iUx+1,iUy)+6*w(iUx,iUy)-4*w(iUx-1,iUy)+w(iUx-2,iUy)...
                 +  w(iUx,iUy+2)-4*w(iUx,iUy+1)+6*w(iUx,iUy)-4*w(iUx,iUy-1)+w(iUx,iUy-2)...
                 +  2.*(w(iUx+1,iUy+1) - 2.*w(iUx+1,iUy) + w(iUx+1,iUy-1) - 2*w(iUx,iUy+1) + 4*w(iUx,iUy) - 2*w(iUx,iUy-1) + w(iUx-1,iUy+1) - 2*w(iUx-1,iUy) + w(iUx-1,iUy-1) ) );

            wNext(iUx,iUy) = (k^2/(1+sig0_plate*k))*( (-K_plate^2/h_plate^4)*fac1 ...
                 + (2*sig1_plate/k)*(1/h_plate^2)*(w(iUx+1,iUy)+w(iUx-1,iUy) + w(iUx,iUy+1) + w(iUx,iUy-1) - 4*w(iUx,iUy) - (wPrev(iUx+1,iUy)+wPrev(iUx-1,iUy) + wPrev(iUx,iUy+1) + wPrev(iUx,iUy-1) - 4*wPrev(iUx,iUy)))...
                 - J_P(iUx,iUy)*F/(rho_plate*H_plate)  ...
                 + (2/k^2)*w(iUx,iUy) - (1-sig0_plate*k)*wPrev(iUx,iUy)/k^2 );

        end    

        vr(t) = vrel;
    else % if model is not bowed but just excited with an impulse of sorts

        I_S_J_S = sum(J_S(:).*J_S(:).* h);     
        I_P_J_P = sum(J_P(:).*J_P(:).* h_plate^2);     
        I_S_J_B = 0;
        I_S_u_n = sum(J_S(:).*u(:).* h);
        I_S_u_n_min_1 = sum(J_S(:).*uPrev(:).* h);
        I_P_w_n = sum(J_P(:).*w(:).* h_plate^2);
        I_P_w_n_min_1 = sum(J_P(:).*wPrev(:).* h_plate^2);

        I_S_dxx_u_n = (1/h^2)*sum(sum(I_S(3:N-2).*(u(4:N-1)-2.*u(3:N-2)+u(2:N-3))));
        I_S_dxx_u_n_min_1 = (1/h^2)*sum(sum(I_S(3:N-2).*(uPrev(4:N-1)-2.*uPrev(3:N-2)+uPrev(2:N-3))));
        I_S_dxxxx_u_n = (1/h^4)*sum(sum(I_S(3:N-2).*(u(5:N)-4.*u(4:N-1)+6.*u(3:N-2)-4.*u(2:N-3)+u(1:N-4))));
%         IS_dxxxx_u_n_min_1 = (1/h^4)*sum(sum(I_S(3:N-2).*(uPrev(5:N)-4.*uPrev(4:N-1)+6.*uPrev(3:N-2)-4.*uPrev(2:N-3)+uPrev(1:N-4))));

        IP_delta_laplace_w_n = (1/h_plate^2)*sum(sum(I_P(:,3:N_plate-2).*(w(:,4:N_plate-1)-2.*w(:,3:N_plate-2)+w(:,2:N_plate-3))))...
                             + (1/h_plate^2)*sum(sum(I_P(3:N_plate-2,:).*(w(4:N_plate-1,:)-2.*w(3:N_plate-2,:)+w(2:N_plate-3,:))));   
        IP_delta_laplace_w_n_min_1 = (1/h_plate^2)*sum(sum(I_P(:,3:N_plate-2).*(wPrev(:,4:N_plate-1)-2.*wPrev(:,3:N_plate-2)+wPrev(:,2:N_plate-3))))...
                                   + (1/h_plate^2)*sum(sum(I_P(3:N_plate-2,:).*(wPrev(4:N_plate-1,:)-2.*wPrev(3:N_plate-2,:)+wPrev(2:N_plate-3,:))));                            
        IP_delta_laplace_x2_w_n = (1/h_plate^4)*sum(sum(I_P(:,3:N_plate-2).*(w(:,5:N_plate)-4.*w(:,4:N_plate-1)+6.*w(:,3:N_plate-2)-4.*w(:,2:N_plate-3)+w(:,1:N_plate-4))))...
                                + (1/h_plate^4)*sum(sum(I_P(3:N_plate-2,:).*(w(5:N_plate,:)-4.*w(4:N_plate-1,:)+6.*w(3:N_plate-2,:)-4.*w(2:N_plate-3,:)+w(1:N_plate-4,:))))...
                                + (2/h_plate^4)*sum(sum(I_P(3:N_plate-2,3:N_plate-2).*( w(4:N_plate-1,4:N_plate-1) - 2.*w(4:N_plate-1,3:N_plate-2) + w(4:N_plate-1,2:N_plate-3)- 2.*w(3:N_plate-2,4:N_plate-1) + 4.*w(3:N_plate-2,3:N_plate-2) - 2.*w(3:N_plate-2,2:N_plate-3) + w(2:N_plate-3,4:N_plate-1) - 2.*w(2:N_plate-3,3:N_plate-2) + w(2:N_plate-3,2:N_plate-3) )));  


        b = (k^2/(1+sig0*k))*(c^2.*I_S_dxx_u_n - K^2.*I_S_dxxxx_u_n + (2*sig1/k)*(I_S_dxx_u_n - I_S_dxx_u_n_min_1)) + ...
            (k^2/(1+sig0*k))*(2/k^2)*I_S_u_n - (k^2/(1+sig0*k))*((1-sig0*k)/k^2)*I_S_u_n_min_1 - ...
            (k^2/(1+sig0_plate*k))*(-K_plate^2.*IP_delta_laplace_x2_w_n + (2*sig1_plate/k)*(IP_delta_laplace_w_n-IP_delta_laplace_w_n_min_1)) - ...
            (k^2/(1+sig0_plate*k))*(2/k^2).*I_P_w_n + (k^2/(1+sig0_plate*k))*((1-sig0_plate*k)/k^2)*I_P_w_n_min_1;

        F = (-b+(k^2/(1+sig0*k))*I_S_J_B*((0)/(rho*A)))/(k^2*I_S_J_S/((1+sig0*k)*rho*A) + k^2*I_P_J_P/((1+sig0_plate*k)*rho_plate*H_plate));

        if ~connection
            F = 0;
        end        

        uNext(3:N-2) = (k^2/(1+sig0*k))*( (c^2/h^2)*(u(4:N-1) - 2*u(3:N-2) + u(2:N-3))...
                        +(2*sig1/(k*h^2))*(u(4:N-1) - 2*u(3:N-2) + u(2:N-3) - uPrev(4:N-1) + 2*uPrev(3:N-2) - uPrev(2:N-3))...
                        - K^2*(1/h^4)*(u(5:N) - 4*u(4:N-1) + 6*u(3:N-2) - 4*u(2:N-3) + u(1:N-4))...
                        + J_S(2)*F/(rho*A) ...                        
                        + (2/k^2)*u(3:N-2) - (1-sig0*k).*uPrev(3:N-2)./k^2  );

        if strcmp(BC_string,'SimplySupported')
            uNext(2) = (k^2/(1+sig0*k))*( (c^2/h^2)*(u(3) - 2*u(2) + u(1))...
                            +(2*sig1/(k*h^2))*(u(3) - 2*u(2) + u(1) - uPrev(3) + 2*uPrev(2) - uPrev(1))...
                            - K^2*(1/h^4)*(u(4) - 4*u(3) + 6*u(2) - 4*u(1) - u(2))...
                            + J_S(2)*F/(rho*A) ...
                            + (2/k^2)*u(2) - (1-sig0*k).*uPrev(2)./k^2  );                    

            uNext(N-1) = (k^2/(1+sig0*k))*( (c^2/h^2)*(u(N) - 2*u(N-1) + u(N-2))...
                            +(2*sig1/(k*h^2))*(u(N) - 2*u(N-1) + u(N-2) - uPrev(N) + 2*uPrev(N-1) - uPrev(N-2))...
                            - K^2*(1/h^4)*(-u(N-1) - 4*u(N) + 6*u(N-1) - 4*u(N-2) + u(N-3))...
                            + J_S(N-1)*F/(rho*A) ...
                            + (2/k^2)*u(N-1) - (1-sig0*k).*uPrev(N-1)./k^2  );
        end                    
                    
        for iAll = 1:length(locsDo)

            iUx = rem(locsDo(iAll)-1,size(wNext,1))+1; % rem is remainder after division. rem(23,5) = 3
            iUy = (locsDo(iAll)-iUx)/size(wNext,1) + 1;


            fac1 = (w(iUx+2,iUy)-4*w(iUx+1,iUy)+6*w(iUx,iUy)-4*w(iUx-1,iUy)+w(iUx-2,iUy)...
                 +  w(iUx,iUy+2)-4*w(iUx,iUy+1)+6*w(iUx,iUy)-4*w(iUx,iUy-1)+w(iUx,iUy-2)...
                 +  2.*(w(iUx+1,iUy+1) - 2.*w(iUx+1,iUy) + w(iUx+1,iUy-1) - 2*w(iUx,iUy+1) + 4*w(iUx,iUy) - 2*w(iUx,iUy-1) + w(iUx-1,iUy+1) - 2*w(iUx-1,iUy) + w(iUx-1,iUy-1) ) );

            wNext(iUx,iUy) = (k^2/(1+sig0_plate*k))*( (-K_plate^2/h_plate^4)*fac1 ...
                 + (2*sig1_plate/k)*(1/h_plate^2)*(w(iUx+1,iUy)+w(iUx-1,iUy) + w(iUx,iUy+1) + w(iUx,iUy-1) - 4*w(iUx,iUy) - (wPrev(iUx+1,iUy)+wPrev(iUx-1,iUy) + wPrev(iUx,iUy+1) + wPrev(iUx,iUy-1) - 4*wPrev(iUx,iUy)))...
                 - J_P(iUx,iUy)*F/(rho_plate*H_plate)  ...
                 + (2/k^2)*w(iUx,iUy) - (1-sig0_plate*k)*wPrev(iUx,iUy)/k^2 );

        end

    end
    
    % % UPDATE STATES:       
    uPrev = u;
    u = uNext;

    wPrev = w;
    w = wNext;    

    % % OUTPUTS
    out_at_bridge(t) = sum(I_S.*uNext);
    out_plate(t) = sum(I_plate_out(:).*wNext(:));
    out_string(t) = sum(uNext.*I_grid_out);
    out_string_2_3rd(t) = sum(uNext.*I_grid_2_3rd_str);
    out_string_1_4th(t) = sum(uNext.*I_grid_1_4th_str);

    F_vec(t) = F;     



    if plotFlag
        if t >= 400
%         if t >= 1

            wPlot = w;
            locsPrint = [1:size(wPlot,1)*size(wPlot,2)];
            locsPrint = setdiff(locsPrint,locsDo);
            wPlot(locsPrint) = NaN;

            vec = [1:length(u)];
            figure(123)
            subplot(2,1,1)
            hold off
            plot(vec.*h,u,'linewidth',2)
            grid on
            hold all
            plot([conn_point*L],... % maybe make it discretized ! 
                 [out_at_bridge(t)],'ro','linewidth',2)  
            if bowed
                plot(bp * L,sum(uNext.*I_B),'go','linewidth',3)
            end
            xlabel('Xdir [m]')
            ylabel('Displacement, u [m]')
            subplot(2,1,2)
            surf(X.*h_plate,Y.*h_plate,wPlot,'linewidth',2)
            hold all
            plot3(x_inp,y_inp,20,'ro','linewidth',3)
            plot3(x_out,y_out,20,'gd','linewidth',3)
            hold off
            shading interp
            colorbar
%             caxis([-3e-6 3e-6])
%             caxis([-3e-5 3e-5])
            view([0 90])    
            axis equal
            xlabel('Xdir [m]')
            ylabel('Ydir [m]')
    %         caxis([0,1e-5])

            pause
        end
    end
    

    
    
% % % % % A bunch of possible plots.. 
% % % %         figure;
% % % %         surf(X.*h_plate,Y.*h_plate,wPlot,'linewidth',2) 
% % % %         hold all
% % % %         surf(X.*h_plate,Y.*h_plate,I_P./1e3)        
% % % %         plot3(X(l_inp,m_inp).*h_plate,Y(l_inp,m_inp).*h_plate,20,'ko','linewidth',2)
% % % %         plot3(X(l_out,m_out).*h_plate,Y(l_out,m_out).*h_plate,20,'go','linewidth',2)
% % % %         hold off
% % % %         shading interp
% % % %         colorbar
% % % %         caxis([-3e-6 3e-6])
% % % %         view([0 90])        
% % 
% % % %     if (t>1000) && mod(t,500) == 0
% % % %         figure(12344)
% % % % %         plot(vr_vec(t-500:t),Fbow_vector(t-500:t).*rho.*H,'o')
% % % %         plot(z_vec(t-500:t),Fbow_vector(t-500:t),'o')
% % % %         xlim([-9e-5,1e-5])
% % % %         ylim([-0.9,0.1])
% % % %         grid on
% % % %         xlabel('Britle displacement z [m]')
% % % %         ylabel('Bow Force [N]');
% % % %     %     pause
% % % %     end



% % % %     if (t>1000) && mod(t,500) == 0
% % % % 
% % % % %         vrel_last_vec = [-2:0.00001:2];
% % % %         vrel_last_vec = [-0.5:0.00001:0.5];
% % % %         z_ss_vec = sign(vrel_last_vec)./s0 .* (fC + (fS - fC)*exp(-(vrel_last_vec./vS).^2));
% % % % 
% % % %         figure(2323);
% % % %         subplot(2,1,1);
% % % %         plot(vrel_last_vec,z_ss_vec,'linewidth',2);
% % % %         grid on
% % % %         hold all
% % % %         plot(vrel,z_ss,'o','linewidth',2);
% % % %         hold off
% % % % 
% % % %         z_last_vec = linspace(-2*z_ss,2*z_ss,1000);
% % % %         z_last = z;
% % % % 
% % % %         alpha_fct_vec = zeros(size(z_last_vec));
% % % %         alpha_fct_correct_vec = zeros(size(z_last_vec));
% % % %         for i = 1:length(z_last_vec)
% % % % 
% % % %             vrel_last = vrel;
% % % %             z_last = z_last_vec(i);
% % % % 
% % % %             if abs(z_last)<=z_ba
% % % %                 alpha_fct_test = 0;
% % % %                 alpha_fct_test_correct = 0;
% % % %             elseif (z_ba < abs(z_last)) && (abs(z_last) < abs(z_ss))
% % % %                 alpha_fct_test = 0.5*(1 + sin(sign(z_last)*(pi*(z_last - sign(z_last)*0.5*(abs(z_ss)+z_ba))/(abs(z_ss)-z_ba))));                
% % % %                 alpha_fct_test_correct = 0.5*(1 + sign(z_last)*sin(pi*(z_last - sign(z_last)*0.5*(abs(z_ss)+z_ba))/(abs(z_ss)-z_ba)));
% % % %             elseif abs(z_last) >= abs(z_ss)
% % % %                 alpha_fct_test = 1;
% % % %                 alpha_fct_test_correct = 1;
% % % %             end
% % % %             alpha_fct_vec(i) = alpha_fct_test;
% % % %             alpha_fct_correct_vec(i) = alpha_fct_test_correct;
% % % %         end
% % % % 
% % % %         figure(2323);
% % % %         subplot(2,1,2);
% % % %         plot(z_last_vec,alpha_fct_vec,'linewidth',2);
% % % %         grid on
% % % %         hold all
% % % %         plot(z_last_vec,alpha_fct_correct_vec,'linewidth',2);        
% % % %         if sign(vrel) == sign(z)
% % % %             plot(z,alpha_fct,'ro','linewidth',2);
% % % %         else
% % % %             plot(z,alpha_fct,'gd','linewidth',2);
% % % %         end
% % % %         hold off
% % % % %         pause
% % % %  
% % % %     end



end
%%%%%% end main loop



% % Saving sounds
out_at_bridge_exp_curve = out_at_bridge.*exp_curve_comb.';
audiowrite(fullfile(pathSave,['sound_amoeba_at_bridge_',bowingModel,'_note_',txt{iFreq,1},'.wav']),...
           out_at_bridge_exp_curve./max(abs(out_at_bridge_exp_curve)),Fs)

out_at_plate_exp_curve = out_plate.*exp_curve_comb.';
audiowrite(fullfile(pathSave,['sound_amoeba_at_plate_',bowingModel,'_note_',txt{iFreq,1},'.wav']),...
           out_at_plate_exp_curve./max(abs(out_at_plate_exp_curve)),Fs)

out_at_string_exp_curve = out_string_2_3rd.*exp_curve_comb.';
audiowrite(fullfile(pathSave,['sound_amoeba_raw_at_string_',bowingModel,'_note_',txt{iFreq,1},'.wav']),...
           out_at_string_exp_curve./max(abs(out_at_string_exp_curve)),Fs)

audiowrite(fullfile(pathSave,['sound_amoeba_raw_at_bridge_',bowingModel,'_note_',txt{iFreq,1},'.wav']),...
           out_at_bridge./max(abs(out_at_bridge)),Fs)

audiowrite(fullfile(pathSave,['sound_amoeba_raw_at_plate_',bowingModel,'_note_',txt{iFreq,1},'.wav']),...
           out_plate./max(abs(out_plate)),Fs)

audiowrite(fullfile(pathSave,['sound_amoeba_raw_at_string_',bowingModel,'_note_',txt{iFreq,1},'_',bowingModel,'.wav']),...
           out_string_2_3rd./max(abs(out_string_2_3rd)),Fs)

  
       
       
figure(5544)
plot(tVec,out_at_bridge);
saveas(figure(5544),fullfile(pathSaveFig,['out_at_bridge_',bowingModel,'_',txt{iFreq,1},'.png']))

figure(5545)
plot(tVec,out_plate);
saveas(figure(5545),fullfile(pathSaveFig,['out_plate_',bowingModel,'_',txt{iFreq,1},'.png']))

figure(5546)
plot(tVec,out_string_2_3rd);
saveas(figure(5546),fullfile(pathSaveFig,['out_string_',bowingModel,'_',txt{iFreq,1},'.png']))


figure(15);
plot(z_vec(1:t))
ylabel('z')
hold all
saveas(figure(15),fullfile(pathSaveFig,['z_',txt{iFreq,1},'.png']))

figure(16);
plot(vr(1:t))
ylabel('vr')
hold all
saveas(figure(16),fullfile(pathSaveFig,['vr_',bowingModel,'_',txt{iFreq,1},'.png']))



figure(65432);
plot(Fbow_vector(1:t),'-')
grid on
hold all
ylabel('Fbow')
saveas(figure(65432),fullfile(pathSaveFig,['FBow_',txt{iFreq,1},'.png']))

figure(6543111);
plot(vr(22000:22500),Fbow_vector(22000:22500),'o','linewidth',2)
grid on
hold all
xlabel('vr [m/s]')
ylabel('Fbow [N]')
saveas(figure(6543111),fullfile(pathSaveFig,['vr_vs_FBow_',bowingModel,'_',txt{iFreq,1},'.png']))



signal = out_plate;
Nwin = length(signal);
KK = Nwin*2;
samples = [0:Nwin-1];
f = (0:KK-1)*Fs/KK;

xdft = fft(signal,KK);
xdft = xdft/Nwin;

figure(76765);
subplot(2,1,1)
plot(f,abs(xdft),'linewidth',2);
grid on
hold all
xlim([10 Fs/2])
set(gca,'xscale','log')
subplot(2,1,2)
plot(f,abs(xdft),'linewidth',2);
grid on
hold all 
xlim([0,20000])
set(gca,'xscale','linear')



Nwin = length(out_at_bridge_exp_curve(25001:30000));
KK = Nwin*2;
samples = [0:Nwin-1];
f = (0:KK-1)*Fs/KK;

xdft = fft(out_at_bridge_exp_curve,KK);
xdft = xdft/Nwin;

figure(7654567);
subplot(2,1,1)
plot(f,abs(xdft),'linewidth',2);
grid on
hold all
xlim([10 Fs/2])
set(gca,'xscale','log')
subplot(2,1,2)
plot(f,abs(xdft),'linewidth',2);
grid on
xlim([0,20000])
set(gca,'xscale','linear')
saveas(figure(7654567),fullfile(pathSaveFig,['out_at_bridge_fft_',bowingModel,'_',txt{iFreq,1},'.png']))


Nwin = length(out_at_plate_exp_curve(25001:40000));
KK = Nwin*2;
samples = [0:Nwin-1];
f = (0:KK-1)*Fs/KK;

xdft = fft(out_at_plate_exp_curve(25001:40000),KK);
xdft = xdft/Nwin;

figure(7654568);
subplot(2,1,1)
plot(f,abs(xdft),'linewidth',2);
grid on
hold all
xlim([0 Fs/2])
set(gca,'xscale','log')
subplot(2,1,2)
plot(f,abs(xdft),'linewidth',2);
grid on
xlim([0,1000])
set(gca,'xscale','linear')
saveas(figure(7654568),fullfile(pathSaveFig,['out_at_plate_fft_',bowingModel,'_',txt{iFreq,1},'.png']))




Nwin = length(out_at_string_exp_curve(25001:40000));
KK = Nwin*2;
samples = [0:Nwin-1];
f = (0:KK-1)*Fs/KK;

xdft = fft(out_at_string_exp_curve(25001:40000),KK);
xdft = xdft/Nwin;

figure(7654569);
subplot(2,1,1)
plot(f,abs(xdft),'linewidth',2);
grid on
hold all
xlim([0 Fs/2])
set(gca,'xscale','log')
subplot(2,1,2)
plot(f,abs(xdft),'linewidth',2);
grid on
xlim([0,1000])
set(gca,'xscale','linear')
saveas(figure(7654569),fullfile(pathSaveFig,['out_at_string_fft_',bowingModel,'_',txt{iFreq,1},'.png']))


M = 256*3;
overlap = round(M*3/4); % overlap
KK = M*2;
win = window(@hann,M);

[stft,freq,time] = spectrogram(out_at_bridge_exp_curve./max(abs(out_at_bridge_exp_curve)),win,overlap,KK,Fs);

figure(892122887);
surf(time,freq,20*log10(abs(stft)))
view([0 90])
shading interp
xlabel('Time [s]')
ylabel('Freq [Hz]')
% xlim([0,1])
ylim([0,20000])
caxis([-150,20])
colorbar
set(gca,'yscale','linear');
saveas(figure(892122887),fullfile(pathSaveFig,['out_at_bridge_spectrogram_',bowingModel,'_',txt{iFreq,1},'.png']))

if connection
[stft,freq,time] = spectrogram(out_at_plate_exp_curve./max(abs(out_at_plate_exp_curve)),win,overlap,KK,Fs);

figure(89899888);
surf(time,freq,20*log10(abs(stft)))
view([0 90])
shading interp
xlabel('Time [s]')
ylabel('Freq [Hz]')
% xlim([0,1])
ylim([0,20000])
caxis([-150,20])
colorbar
set(gca,'yscale','linear');
saveas(figure(89899888),fullfile(pathSaveFig,['out_at_plate_spectrogram_',bowingModel,'_',txt{iFreq,1},'.png']))
end



[stft,freq,time] = spectrogram(out_at_string_exp_curve./max(abs(out_at_string_exp_curve)),win,overlap,KK,Fs);

figure(89899889);
surf(time,freq,20*log10(abs(stft)))
view([0 90])
shading interp
xlabel('Time [s]')
ylabel('Freq [Hz]')
% xlim([0,1])
ylim([0,20000])
caxis([-150,20])
colorbar
set(gca,'yscale','linear');
saveas(figure(89899889),fullfile(pathSaveFig,['out_at_string_spectrogram_',bowingModel,'_',txt{iFreq,1},'.png']))


% close all





