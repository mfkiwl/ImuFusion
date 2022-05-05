%clear; close all; clc;

% --- Dataset parameters
g = 9.81;
sigma_acc  = 0.82e-2;           % accelerometer noise (m/s^2)
sigma_gyro = 0.88e-3;    % gyroscope noise (rad/s)
sigma_vel = 0.01;           % zero-velocity update measurement noise (m/s)

%% zero-velocity detector parameters
cova  = 0.01^2; 
covw  = (0.1*pi/180)^2;     
W     = 5; % window size
gamma  = 0.3e5;

N = size(imu, 1);

%% Since we have got all the data, we can run the zero velocity detection
%%  for the whole dataset.
iszv = zeros(1, N);
T = zeros(1, N-W+1);
for k = 1:N-W+1
    mean_a = mean(imu(k:k+W-1,2:4));
    for l = k:k+W-1
        temp = imu(l,2:4) - g * mean_a / norm(mean_a);
        T(k) = T(k) + imu(l,5:7)*imu(l,5:7)'/covw + temp*temp'/cova;
    end
end
T = T./W;
for k = 1:size(T,2)
    if T(k) < gamma
        iszv(k:k+W-1) = ones(1,W);
    end
end

init_a = mean(imu(300:900,2:4));
init_a =-init_a / norm(init_a);

init_psi =  0; % yaw 
init_theta = -asin(init_a(1)); % pitch
init_phi = atan2(init_a(2), init_a(3)); % roll

init_quat = angle2quat(init_psi, init_theta, init_phi);

% Estimate sensor bias.
Rsw = quat2dcm(init_quat);
as  = Rsw * [0;0;-g];
bias_a = mean(imu(300:900,2:4)) - as';
bias_w = mean(imu(300:900,5:7));

% set the initial state vector
x = zeros(10,1);
x(7:10,1) = init_quat';

% set the initial covariance
P = diag([1e-10*ones(1,6), 1e-6*ones(1,4)]);

%
x_r = zeros(10,N);
x_r(:,1) = x;

% measurement matrix
H = [zeros(3), eye(3), zeros(3,4)];
R =  sigma_vel^2 * eye(3);

%%
%=========================================================================%
%==                             Main  Loop                               =%
%=========================================================================%
for k = 2:N
    %% compute state transition matrix F and covariance Q
    dt = imu(k,1)-imu(k-1,1);
    w = imu(k-1,5:7)*0.5+imu(k,5:7)*0.5; % - bias_w;
    quat = x(7:10,1);
    a = imu(k,2:4); % - bias_a;
    
    % continuous state transition matrix
    Ow = [0     -w(1)   -w(2)    -w(3);...
          w(1)   0       w(3)    -w(2);...
          w(2)  -w(3)    0        w(1);...
          w(3)   w(2)   -w(1)     0  ];
    Vq = compVq(quat, a);
    
    Fc = zeros(10);
    Fc(1:3, 4:6) = eye(3);
    Fc(4:10,7:10)= [Vq; 0.5*Ow];
    
    % continuous process covariance
    Gq = 0.5* [-quat(2)  -quat(3)   -quat(4); ...
                quat(1)  -quat(4)    quat(3); ...
                quat(4)   quat(1)   -quat(2); ...
               -quat(3)   quat(2)    quat(1)];       
    Qc = zeros(10);
    Qc(4:6, 4:6)  =  sigma_acc^2*eye(3);
    Qc(7:10,7:10) =  sigma_gyro^2*(Gq*Gq');
    
    % discretilization
    F = eye(10) + Fc* dt;
    Q = Qc* dt;
    
    %% state propagation
    R_S_n = quat2dcm(quat');
    acc = R_S_n' * a' + [0; 0;  g];
    
    x(1:3) = x(1:3) + x(4:6)* dt + 0.5*acc* dt^2;
    x(4:6) = x(4:6) + acc* dt;
    
    quat = (eye(4) + 0.5*Ow* dt)*quat;
    quat = quat/norm(quat);
    x(7:10) = quat;
    
    %% covariance propagation
    P = F*P*F' + Q;

    %% zero-velocity update
    if iszv(k) == 1
        K = (P*H')/(H*P*H'+R);
        y = -x(4:6);
        
        x = x + K*y;
        x(7:10) = x(7:10)/norm(x(7:10));
        
        P = (eye(10)-K*H)*P;
    end
    
    P = (P+P')/2;
    
    x_r(:,k) = x;
      
end

% yaw, pitch, roll 
[psi, theta, phi] = quat2angle(x_r(7:10,:)');

plot([psi, theta, phi], '-','marker','.');    

