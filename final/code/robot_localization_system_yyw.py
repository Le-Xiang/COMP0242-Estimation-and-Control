#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def wrap_angle(angle): 
    return np.arctan2(np.sin(angle), np.cos(angle))

class FilterConfiguration(object):
    def __init__(self):
        # Process and measurement noise covariance matrices
        self.V = np.diag([0.1, 0.1, 0.05]) ** 2  # Process noise covariance
        # Measurement noise variance (range measurements) original value is 0.5
        self.W_range = 1 ** 2
        self.W_bearing = (np.pi * 1 / 180.0) ** 2

        # Initial conditions for the filter
        self.x0 = np.array([2.0, 3.0, np.pi / 4])
        self.Sigma0 = np.diag([1.0, 1.0, 0.5]) ** 2


class Map(object):
    def __init__(self):
        self.landmarks = np.array([
            [5, 10],
            [15, 5],
            [10, 15]
        ])
        # self.landmarks = np.array([
        #     [x, y] for x in np.linspace(-40, 60, 10)
        #             for y in np.linspace(-40, 60, 10)
        # ])



class RobotEstimator(object):

    def __init__(self, filter_config, map):
        # Variables which will be used
        self._config = filter_config
        self._map = map

    # This nethod MUST be called to start the filter
    def start(self):
        self._t = 0
        self._set_estimate_to_initial_conditions()

    def set_control_input(self, u):
        self._u = u

    # Predict to the time. The time is fed in to
    # allow for variable prediction intervals.
    def predict_to(self, time):
        # What is the time interval length?
        dt = time - self._t

        # Store the current time
        self._t = time

        # Now predict over a duration dT
        self._predict_over_dt(dt)

    # Return the estimate and its covariance
    def estimate(self):
        return self._x_est, self._Sigma_est

    # This method gets called if there are no observations
    def copy_prediction_to_estimate(self):
        self._x_est = self._x_pred
        self._Sigma_est = self._Sigma_pred

    # This method sets the filter to the initial state
    def _set_estimate_to_initial_conditions(self):
        # Initial estimated state and covariance
        self._x_est = self._config.x0
        self._Sigma_est = self._config.Sigma0

    # Predict to the time
    def _predict_over_dt(self, dt):
        v_c = self._u[0]
        omega_c = self._u[1]
        V = self._config.V

        # Predict the new state
        self._x_pred = self._x_est + np.array([
            v_c * np.cos(self._x_est[2]) * dt,
            v_c * np.sin(self._x_est[2]) * dt,
            omega_c * dt
        ])
        self._x_pred[-1] = np.arctan2(np.sin(self._x_pred[-1]),
                                      np.cos(self._x_pred[-1]))

        # Predict the covariance
        A = np.array([
            [1, 0, -v_c * np.sin(self._x_est[2]) * dt],
            [0, 1,  v_c * np.cos(self._x_est[2]) * dt],
            [0, 0, 1]
        ])

        self._kf_predict_covariance(A, self._config.V * dt)

    # Predict the EKF covariance; note the mean is
    # totally model specific, so there's nothing we can
    # clearly separate out.
    def _kf_predict_covariance(self, A, V):
        self._Sigma_pred = A @ self._Sigma_est @ A.T + V

    # Implement the Kalman filter update step.
    def _do_kf_update(self, nu, C, W):

        # Kalman Gain
        SigmaXZ = self._Sigma_pred @ C.T
        SigmaZZ = C @ SigmaXZ + W
        K = SigmaXZ @ np.linalg.inv(SigmaZZ)

        # State update
        self._x_est = self._x_pred + K @ nu


        # Covariance update
        self._Sigma_est = (np.eye(len(self._x_est)) - K @ C) @ self._Sigma_pred



    def update_with_range_bearing(self, y_range_bearing):
        """使用距离-方位测量数据更新状态估计"""
        y_pred = []
        C = []
        x_pred = self._x_pred

        for lm in self._map.landmarks:
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)

            bearing_pred = np.arctan2(dy_pred, dx_pred) - x_pred[2]
            # 包裹角度，确保方位角在 [-π, π] 范围内
            # bearing_pred = wrap_angle(bearing_pred)
            # 将预测的距离和方位角添加到预测列表中
            y_pred.append([range_pred, bearing_pred])


            # 计算测量模型的雅可比矩阵 C
            C_range_bearing = np.array([
                [-(dx_pred) / range_pred, -(dy_pred) / range_pred, 0],  # 距离的雅可比
                [dy_pred / (range_pred**2), -dx_pred / (range_pred**2), -1]  # 方位角的雅可比
            ])
            C.append(C_range_bearing)

        # 转换为数组
        C = np.array(C)
        # print(C.shape)
        C = np.vstack(C)
        # print(C.shape)
        y_pred = np.array(y_pred)

        # 计算创新量（测量与预测的差值）
        # print(f"y_range_bearing shape is:{y_range_bearing.shape}")
        # print(f"y_pred shape is :{y_pred.shape}")
        nu = y_range_bearing - y_pred
        # 对方位角进行角度包裹，确保在 [-π, π] 范围内
        nu[:, 1] = wrap_angle(nu[:, 1])
        # print(nu.shape)
        # nu = nu.flatten()
        nu = np.hstack(nu)
        # 构建测量协方差矩阵
        W_landmarks = np.block([
            [self._config.W_range * np.eye(len(self._map.landmarks)), np.zeros((len(self._map.landmarks), len(self._map.landmarks)))],
            [np.zeros((len(self._map.landmarks), len(self._map.landmarks))), self._config.W_bearing * np.eye(len(self._map.landmarks))]
        ])

        # 使用卡尔曼滤波器的更新步骤
        self._do_kf_update(nu, C, W_landmarks)

        # 更新后的状态角度也需要包裹
        self._x_est[-1] = wrap_angle(self._x_est[-1])



    def update_from_landmark_range_observations(self, y_range):

        # Predicted the landmark measurements and build up the observation Jacobian
        y_pred = []
        C = []
        x_pred = self._x_pred
        for lm in self._map.landmarks:
            dx_pred = lm[0] - x_pred[0]
            dy_pred = lm[1] - x_pred[1]
            range_pred = np.sqrt(dx_pred**2 + dy_pred**2)
            y_pred.append(range_pred)

            # Jacobian of the measurement model
            C_range = np.array([
                -(dx_pred) / range_pred,
                -(dy_pred) / range_pred,
                0
            ])
            C.append(C_range)
        # Convert lists to arrays
        C = np.array(C)
        print(C.shape)
        y_pred = np.array(y_pred)

        # Innovation. Look new information! (geddit?)
        nu = y_range - y_pred
        # Since we are oberving a bunch of landmarks
        # build the covariance matrix. Note you could
        # swap this to just calling the ekf update call
        # multiple times, once for each observation,
        # as well
        W_landmarks = self._config.W_range * np.eye(len(self._map.landmarks))
        self._do_kf_update(nu, C, W_landmarks)

        # Angle wrap afterwards
        self._x_est[-1] = np.arctan2(np.sin(self._x_est[-1]),
                                     np.cos(self._x_est[-1]))
