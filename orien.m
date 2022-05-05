%clear; close all; clc;

%--- Dataset parameters
noise_gyro = 0.88e-3;        % Gyroscope noise(discrete), rad/s
noise_accel = 0.82e-2;      % Accelerometer noise, m/s^2
gravity = 9.81007;          % Gravity magnitude, m/s^2

bias_w = mean(imu(300:900, 5:7));    % gyroscope bias
bias_a = mean(imu(300:900, 2:4));    % accelerometer bias

%--- Container of the results
N = size(imu,1);
allX = zeros(N, 4);


%--- Initialization
x = [1 0 0 0]';            % Initial state (quaternion)
P = 1e-10 * eye(4);         % Initial covariance
allX(1,:) = x';

% ---Here we go !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for k = 2 : N
    
    %--- 1. Propagation --------------------------
    % Gyroscope measurements
    dt = imu(k,1)-imu(k-1,1);
    w = (imu(k-1, 5:7) + imu(k, 5:7))/2;
    w = w - bias_w;
    
    % Compute the F matrix
    Omega =[0     -w(1)   -w(2)   -w(3);...
            w(1)  0       w(3)    -w(2); ...
            w(2)  -w(3)   0        w(1); ...
            w(3)  w(2)    -w(1)    0  ];
    F = eye(4) + dt * Omega / 2;
    
    % Compute the process noise Q
    G = [-x(2)  -x(3)  -x(4); ...
          x(1)  -x(4)   x(3); ...
          x(4)   x(1)  -x(2); ...
         -x(3)   x(2)   x(1)] / 2;
    Q = (noise_gyro * dt)^2 * (G * G');
    
    % Propagate the state and covariance
    x = F * x;
    x = x / norm(x);    % Normalize the quaternion
    P = F * P * F' + Q;
    
    
    %--- 2. Update----------------------------------
    % Accelerometer measurements
    a = imu(k, 2:4);
    %a = a - bias_a;
        
    % We use the unit vector of acceleration as observation
    ea = -a' / norm(a);
    ea_prediction = [2*(x(2)*x(4)-x(1)*x(3)); ...
                     2*(x(3)*x(4)+x(1)*x(2)); ...
                     x(1)^2-x(2)^2-x(3)^2+x(4)^2];
    
    % Residual
    y = ea - ea_prediction;
    
    % Compute the measurement matrix H
    H = 2*[-x(3)    x(4)    -x(1)   x(2); ...
            x(2)    x(1)     x(4)   x(3); ...
            x(1)   -x(2)    -x(3)   x(4)];
        
    % Measurement noise R
    R_internal = (noise_accel / norm(a))^2 * eye(3);
    R_external = (1-gravity/norm(a))^2 * eye(3);
    R = R_internal + R_external;
    
    % Update
    S = H * P * H' + R;
    K = P * H' / S;
    x = x + K * y;
    P = (eye(4) - K * H) * P;
    
    
    % 3. Ending
    x = x / norm(x);    % Normalize the quaternion
    P = (P + P') / 2;   % Make sure that covariance matrix is symmetric
    allX(k,:) = x';     % Save the result
    
end

% yaw, pitch, roll 
[psi, theta, phi] = quat2angle(allX);

plot([psi, theta, phi], '-','marker','.');    
