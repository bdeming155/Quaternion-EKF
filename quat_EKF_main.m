sclear
n = 13; % number of states
m = 24; % 8 markers, x,y,z position of each marker = 24 measurements
num_markers = 8; % number of markers
dt = 0.1; % time between measurements

zs = load('p1n00'); % load the noisey marker measurements
[num_meas,~] = size(zs); % number of measurements in the time series

% x,y,z offset between each marker and COM
markers_offset = [-1.5 -3 -4.5; ...
                  -1.5 -3 1.5; ...
                  -1.5 1 -4.5; ...
                  -1.5 1 1.5; ...
                  0.5 -3 -4.5; ...
                  0.5 -3 1.5; ...
                  0.5 1 -4.5; ...
                  0.5 1 1.5];

I_body = diag([1.571428571428571 5.362637362637362 7.065934065934067]);

% compute the symbolic measurement and process models
[h_sym, H_sym] = compute_symbolic_H(markers_offset');
[f_sym, F_sym] = compute_symbolic_F(dt, I_body);

%--------------------------------------------------------------------------
% Initialize
%--------------------------------------------------------------------------
x = hack_init_estimates(zs); % initialize the state vector

%P = 5e-2*eye(n);
P = eye(n);
% P(1,1) = 1;
% P(2,2) = 1;
% P(3,3) = 1;
P(4,4) = 1e-2;
P(5,5) = 1e-2;
P(6,6) = 1e-2;
% P(7,7) = 1;
% P(8,8) = 1;
% P(9,9) = 1;
% P(10,10) = 1;
% P(11,11) = 1;
%P(12,12) = 1;
%P(13,13) = 1;

% Process Covariance
Q = diag([1e-6 1e-6 1e-6 1e-10 1e-10 1e-10 1e-3 1e-3 1e-3 1e-3 1e-3 1e-3 1e-3])...
    *diag(ones(n,1));

% Measurement Covariance
R = 0.36*diag(ones(num_markers*3, 1)); 

% No control input
B = zeros(n,n);
u = zeros(n,1); 

% Initialize the KalmanFilter object with initial conditions
object = KalmanFilter(n, m, x, P, F_sym, B, H_sym, Q, R);

%--------------------------------------------------------------------------
% Main loop
%--------------------------------------------------------------------------

% initialize variables to store the filter estimates
state_estimate = zeros(n, num_meas);
residual = zeros(m, num_meas);
euler_angles = zeros(3,num_meas);
filter_variance = zeros(n, num_meas);

for i = 1:num_meas
    
    % evaluate F, f at every new time step
    [f,object.F] = evaluate_F(object.x, F_sym, f_sym);
    
    % Predict state
    object = object.PredictEKF(u,f);
    
    % Normalize the quaternions in predicted state
    object.x_pred(7:10) = object.x_pred(7:10)/norm(object.x_pred(7:10));
    
    % evaluate H, h at every new step
    [h,object.H] = evaluate_H(object.x_pred, H_sym, h_sym);
    
    % Update state
    [object, resid] = object.UpdateEKF(zs(i,:)',h);
    
    % Normalize Quaternion in state
    object.x(7:10) = object.x(7:10)/norm(object.x(7:10));
    
    % append the results
    state_estimate(:,i) = object.x;
    residual(:,i) = resid;
    euler_angles(:,i) = quat2eul(object.x(7:10)');
    for g = 1:n
        filter_variance(g,i) = object.P(g,g);
    end
end

% plot position estimates
figure(1)
plot(state_estimate(1,1:i))
hold on
plot(state_estimate(2,1:i))
plot(state_estimate(3,1:i))
title('Position filter estimates')
legend('x', 'y', 'z')
xlabel('time (sec)')
ylabel('meters')
grid on
saveas(gcf,'pos_est.jpg')

figure(2)
plot(state_estimate(4,1:i))
hold on
plot(state_estimate(5,1:i))
plot(state_estimate(6,1:i))
title('Velocity filter estimates')
legend('x', 'y', 'z')
xlabel('time (sec)')
ylabel('m/s')
grid on
saveas(gcf,'vel_est.jpg')

figure(3)
plot(state_estimate(7,1:i))
hold on
plot(state_estimate(8,1:i))
plot(state_estimate(9,1:i))
plot(state_estimate(10,1:i))
title('Quaternion filter estimates')
legend('q0', 'q1', 'q2', 'q3')
ylabel('time (sec)')
grid on
saveas(gcf,'quat_est.jpg')

figure(4)
plot(state_estimate(11,1:i))
hold on
plot(state_estimate(12,1:i))
plot(state_estimate(13,1:i))
title('Angular velocity filter estimates')
legend('x', 'y', 'z')
xlabel('time (sec)')
ylabel('rad/sec')
grid on
saveas(gcf,'ang_vel_est.jpg')

figure(5)
plot(euler_angles(1,1:i))
hold on
plot(euler_angles(2,1:i))
plot(euler_angles(3,1:i))
title('Euler Angles filter estimates')
legend('x', 'y', 'z')
xlabel('time (sec)')
ylabel('rad')
grid on
saveas(gcf,'euler_est.jpg')

% Variance Estimates
figure(6)
plot(filter_variance(1,1:i))
hold on
plot(filter_variance(2,1:i))
plot(filter_variance(3,1:i))
title('Position variance estimates')
legend('x', 'y', 'z')
xlabel('time (sec)')
ylabel('m^2')
grid on
saveas(gcf,'pos_var.jpg')

figure(7)
plot(filter_variance(4,1:i))
hold on
plot(filter_variance(5,1:i))
plot(filter_variance(6,1:i))
title('Velocity variance estimates')
xlabel('time (sec)')
ylabel('(m/s)^2')
legend('x', 'y', 'z')
grid on
saveas(gcf,'vel_var.jpg')

figure(8)
plot(filter_variance(7,1:i))
hold on
plot(filter_variance(8,1:i))
plot(filter_variance(9,1:i))
plot(filter_variance(10,1:i))
title('Quaternion variance estimates')
legend('q0', 'q1', 'q2', 'q3')
xlabel('time (sec)')
grid on
saveas(gcf,'quat_var.jpg')

figure(9)
plot(filter_variance(11,1:i))
hold on
plot(filter_variance(12,1:i))
plot(filter_variance(13,1:i))
title('Angular velocity variance estimates')
legend('x', 'y', 'z')
xlabel('time (sec)')
ylabel('(rad/s)^2')
grid on
saveas(gcf,'ang_vel_var.jpg')

marker_1_est = com2marker(state_estimate,markers_offset,num_meas);

figure(10)
plot(marker_1_est(1,:))
hold on
plot(zs(:,1))
title('Marker 1 estimate vs measurement')
legend('estimate', 'measurement')
xlabel('time (sec)')
ylabel('meters')
grid on
saveas(gcf,'marker_1_x_pos.jpg')

function marker_pos = com2marker(state_est,markers_offset,num_meas)
% FUNCTION USEAGE:
% Compute the filter estimate of marker locations from the state estimate of
% COM. Used to evaluate filter performance. Compare this to the raw
% measurements. 
% marker_pos = COM_pos - R*offset

% INPUTS:
% state_est (13xnum_meas) - state estimate of COM from filter.
% markers_offset (8x3) - offset of each marker from COM as matrix. Rows
% correspond to each marker, columns correspond to x, y, z
% num_meas (scalar) - number of measurements in the time series

% OUTPUTS:
% marker_pos (3xnum_meas) - marker 1 location estimate from COM state
% estimate

marker_pos = zeros(3,num_meas);
for f = 1:num_meas
    R = quat2rotm(state_est(7:10,f)');
    marker_pos(:,f) = state_est(1:3,f) - R*markers_offset(1,:)';
end
end

function [f_eval, F_eval] = evaluate_F(state, F_sym, f_sym)
% FUNCTION DESCRIPTION:
% evaluates process model f and its jacobian F at the current time step

% INPUTS:
% state (13x1) - current state estimate
% F_sym (13x13) - symbolic jacobian matrix
% f_sym (13x1) - symbolic process model

% OUTPUTS:
% f_eval (13x1) - process model evaluated at current state
% F_eval (13x13) - jacobian of process model evaluated at curent state

syms  x y z v1 v2 v3 q0 q1 q2 q3 w1 w2 w3 
sym_state = [x; y; z; v1; v2; v3; q0; q1; q2; q3; w1; w2; w3];

% evaluate the nonlinear equations at current step for state prediction
f_eval = double(subs(f_sym, sym_state, state));

% evaluate the jacobian at current step for P_bar (filter covar prediction)
F_eval = double(subs(F_sym, sym_state, state));
end

function [h_eval, H_eval] = evaluate_H(state, H_sym, h_sym)
% FUNCTION DESCRIPTION:
% evaluates measurement model h and its jacobian H at the current time step

% INPUTS:
% state (13x1) - next step state prediction
% H_sym (13x13) - symbolic jacobian matrix
% h_sym (13x1) - symbolic measurement model

% OUTPUTS:
% h_eval (13x1) - measurement model evaluated at predicted state
% H_eval (13x13) - jacobian of measurement model evaluated at predicted state

syms  x y z v1 v2 v3 q0 q1 q2 q3 w1 w2 w3 
sym_state = [x; y; z; v1; v2; v3; q0; q1; q2; q3; w1; w2; w3];

% evaluate nonlinear equations at current step for use in residual calc
h_eval = double(subs(h_sym, sym_state, state));

% evaluate the jacobian for P_hat (filter covariance update)
H_eval = double(subs(H_sym, sym_state, state));
end

function [h, H] = compute_symbolic_H(markers_offset)
% FUNCTION DESCRIPTION:
% Set up symbolic measurment model.  h is a function that computes the
% estimated marker location from the state estimate of the center of mass.
% We need this conversion in order to compute the measurement residual.  H
% is the jacobian of h and is used to compute the kalman gain and the
% filter covariance P. We do this symbolically for two reasons.  The first
% is that I am leveraging matlabs symbolic toolbox to compute the
% complicated jacobian H.  Secondly, we are linearizing at each new
% timestep which involves substituting the current state into the symbolic
% matrix.  Doing this symbolic substitution is convenient.

% INPUTS:
% markers_offset (8x3) - offset between markers and COM

% OUTPUTS:
% h (24x1) - symbolic measurement model. Number of rows correspond to x, y,
% z of each of the 8 markers. For example the first row is the function
% that converts the state estimate to the x axis of marker 1 position.
% H (24x13) - jacobian of h. The entries are the partial dereivatives of the
% measurement function with respect to the state variables

num_markers = 8;
syms q0 q1 q2 q3 x y z w1 w2 w3 v1 v2 v3 x_com y_com z_com
state = [x; y; z; v1; v2; v3; q0; q1; q2; q3; w1; w2; w3];

%eq. 1:
% z = [x,y,z]_COM - R*[x,y,z]_offset
% R is the rotation matrix in terms of quaternions
R = [q0^2 + q1^2 - q2^2 - q3^2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2); ...
     2*(q1*q2 + q0*q3), q0^2 - q1^2 + q2^2 - q3^2, 2*(q2*q3 - q0*q1); ...
     2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0^2 - q1^2 - q2^2 + q3^2];

h = sym('h',[24,1]);

% multiply out equation 1 to populate h
% h is 24x1 vector (x,y,z for each of 8 markers)
marker_ind = 1;
for i = 1:3:3*num_markers
    h(i) = state(1) - R(1,:)*markers_offset(:,marker_ind);
    h(i+1) = state(2) - R(2,:)*markers_offset(:,marker_ind);
    h(i+2) = state(3) - R(3,:)*markers_offset(:,marker_ind);
    marker_ind = marker_ind +1;
end

H = jacobian(h, state);
end

function [f, F] = compute_symbolic_F(dt, I_body)
% FUNCTION DESCRIPTION:
% This function sets up the symbolic process model f and computes the
% symbolic jacobian F. A constant velocity model is used for the position
% and velocity predictions.  The relationship between angular velocity and
% quaternion is used to predict the future quaternion states.  Lastly, the
% relationship between angular acceleration, moment of inertia, and angular
% velocity is used to predict the the future angular velocity state.

% INPUTS:
% dt (scalar) - time between predictions
% I_body (3x3) - moment of inertia matrix

% OUTPUTS:
% f (13x1) - symbolic process model
% F (13x13) - symbolic jacobian of process model

syms  x y z v1 v2 v3 q0 q1 q2 q3 w1 w2 w3
state = [x, y, z, v1, v2, v3, q0, q1, q2, q3, w1, w2, w3];

f = sym('f',[13,1]);

% states 1:6 constant velocity model
f(1:3) = [x + dt*v1;...
          y + dt*v2; ...
          z + dt*v3];
      
f(4:6) = [v1; v2; v3];

% assign the q-dots from paper
f(7) = q0 + dt*0.5*(-q1*w1 - q2*w2 - q3*w3);
f(8) = q1 + dt*0.5*(q0*w1 - q3*w2 + q2*w3);
f(9) = q2 + dt*0.5*(q3*w1 + q0*w2 - q1*w3);
f(10) = q3 + dt*0.5*(-q2*w1 + q1*w2 + q0*w3);

% w model
R = [q0^2 + q1^2 - q2^2 - q3^2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2); ...
     2*(q1*q2 + q0*q3), q0^2 - q1^2 + q2^2 - q3^2, 2*(q2*q3 - q0*q1); ...
     2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0^2 - q1^2 - q2^2 + q3^2];
 
I_w = R*I_body*R';
w_cross = [0 -w3 w2; w3 0 -w1; -w2 w1 0];
f(11:13) = [w1; w2; w3] + dt*inv(I_w)*(w_cross*(I_w*[w1; w2; w3]));
%f(11:13) = [w1; w2; w3];

F = jacobian(f,state);
end


