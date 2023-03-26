classdef KalmanFilter
    properties
        n
        m
        x_pred
        P_pred
        x
        P
        F
        B
        H
        Q
        R
    end
    
    methods
        function obj = KalmanFilter(n, m, x, P, F, B, H, Q, R)
            % Initialize
            obj.n = n; % num states
            obj.m = m; % num measurements
            obj.x = x; % state
            obj.P = P; % state covar
            obj.F = F; % state transition mat
            obj.B = B; % control matrix
            obj.H = H; % measurement model
            obj.Q = Q; % process covar
            obj.R = R; % measurement covar
        end
        
        function [obj, y] = Update(obj, z)
            y = z - obj.H * obj.x; % residual
            K = obj.P_pred * obj.H' * inv(obj.H * obj.P_pred * obj.H' + obj.R); % kalman gain
            obj.x = obj.x_pred + K * y;
            obj.P = (eye(obj.n) - K * obj.H) * obj.P_pred;
        end
        
        function [obj, y] = UpdateEKF(obj, z, h)
            y = z - h; % residual
            K = obj.P_pred * obj.H' * inv(obj.H * obj.P_pred * obj.H' + obj.R); % kalman gain
            obj.x = obj.x_pred + K * y;
            obj.P = (eye(obj.n) - K * obj.H) * obj.P_pred;
        end
        
        function obj = Predict(obj, u)
            obj.x_pred = obj.F * obj.x + obj.B * u;
            obj.P_pred = obj.F * obj.P * obj.F' + obj.Q;
        end
        
        function obj = PredictEKF(obj, u, f)
            obj.x_pred = f;
            obj.P_pred = obj.F * obj.P * obj.F' + obj.Q;
        end
    end
end