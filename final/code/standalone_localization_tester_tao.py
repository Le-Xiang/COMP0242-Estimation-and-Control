#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from robot_localization_system_tao import FilterConfiguration, Map, RobotEstimator

# Simulator configuration; this only contains
# stuff relevant for the standalone simulator.

def wrap_angle(angle): 
    return np.arctan2(np.sin(angle), np.cos(angle))

class SimulatorConfiguration(object):
    def __init__(self):
        self.dt = 0.1
        self.total_time = 1000
        self.time_steps = int(self.total_time / self.dt)

        # Control inputs (linear and angular velocities)
        self.v_c = 1.0  # Linear velocity [m/s]
        self.omega_c = 0.1  # Angular velocity [rad/s]

# Placeholder for a controller.


class Controller(object):
    def __init__(self, config):
        self._config = config

    def next_control_input(self, x_est, Sigma_est):
        return [self._config.v_c, self._config.omega_c]

# This class implements a simple simulator for the unicycle
# robot seen in the lectures.


class Simulator(object):

    # Initialize
    def __init__(self, sim_config, filter_config, map):
        self._config = sim_config
        self._filter_config = filter_config
        self._map = map

    # Reset the simulator to the start conditions
    def start(self):
        self._time = 0
        self._x_true = np.random.multivariate_normal(self._filter_config.x0,
                                                     self._filter_config.Sigma0)
        self._u = [0, 0]

    def set_control_input(self, u):
        self._u = u

    # Predict the state forwards to the next timestep
    def step(self):
        dt = self._config.dt
        v_c = self._u[0]
        omega_c = self._u[1]
        v = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=self._filter_config.V * dt)
        self._x_true = self._x_true + np.array([
            v_c * np.cos(self._x_true[2]) * dt,
            v_c * np.sin(self._x_true[2]) * dt,
            omega_c * dt
        ]) + v
        self._x_true[-1] = np.arctan2(np.sin(self._x_true[-1]),
                                      np.cos(self._x_true[-1]))
        self._time += dt
        return self._time
    
    # Generate range-bearing observations to the landmarks
    def landmark_range_bearing_observations_new(self):
        """生成距离-方位测量数据"""
        y = []
        W_range = self._filter_config.W_range  # 距离测量噪声
        W_bearing = self._filter_config.W_bearing  # 方位角测量噪声

        for lm in self._map.landmarks:
            # 计算地标与机器人之间的相对距离和方位角
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            bearing_true = np.arctan2(dy, dx) - self._x_true[2]
            bearing_true = wrap_angle(bearing_true)

            # 添加测量噪声
            range_meas = range_true + np.random.normal(0, np.sqrt(W_range))
            bearing_meas = bearing_true + np.random.normal(0, np.sqrt(W_bearing))

            # 角度包裹，确保方位角在 [-π, π] 范围内
            bearing_meas = wrap_angle(bearing_meas)

            # 将测量值（距离和方位角）添加到观测列表中
            y.append([range_meas, bearing_meas])
        # print(y)
        y = np.array(y)
        # print(y.shape)
        
        return y
    # Generate range-only observations to the landmarks
    # Get the observations to the landmarks. Return None if none visible
    def landmark_range_observations(self):
        y = []
        C = []
        W = self._filter_config.W_range
        for lm in self._map.landmarks:
            # True range measurement (with noise)
            dx = lm[0] - self._x_true[0]
            dy = lm[1] - self._x_true[1]
            range_true = np.sqrt(dx**2 + dy**2)
            range_meas = range_true + np.random.normal(0, np.sqrt(W))
            y.append(range_meas)

        y = np.array(y)
        return y

    def x_true(self):
        return self._x_true


# Create the simulator configuration.
sim_config = SimulatorConfiguration()

# Create the filter configuration. If you want
# to investigate mis-tuning the filter,
# create a different filter configuration for
# the simulator and for the filter, and
# change the parameters between them.
filter_config = FilterConfiguration()

# Create the map object for the landmarks.
map = Map()

# Create the controller. This just provides
# fixed control inputs for now.
controller = Controller(sim_config)

# Create the simulator object and start it.
simulator = Simulator(sim_config, filter_config, map)
simulator.start()

# Create the estimator and start it.
estimator = RobotEstimator(filter_config, map)
estimator.start()

# Extract the initial estimates from the filter
# (which are the initial conditions) and use
# these to generate the control for the first timestep.
x_est, Sigma_est = estimator.estimate()
u = controller.next_control_input(x_est, Sigma_est)

# Arrays to store data for plotting
x_true_history = []
x_est_history = []
Sigma_est_history = []

# Main loop
for step in range(sim_config.time_steps):

    # Set the control input and propagate the
    # step the simulator with that control iput.
    simulator.set_control_input(u)
    simulation_time = simulator.step()

    # Predict the Kalman filter with the same
    # control inputs to the same time.
    estimator.set_control_input(u)
    estimator.predict_to(simulation_time)

    # Use this to get the landmark observations for range-only measurements.
    # y = simulator.landmark_range_observations()

    # Alternatively, use this to get the landmark observations for range-bearing measurements.
    y = simulator.landmark_range_bearing_observations_new()

    # Use this to update the filter with the latest observations for range-only measurements.
    # estimator.update_from_landmark_range_observations(y)

    # Alternatively, use this to update the filter with the latest observations for range-bearing measurements.
    estimator.update_with_range_bearing(y)

    # Get the current state estimate.
    x_est, Sigma_est = estimator.estimate()

    # Figure out what the controller should do next.
    u = controller.next_control_input(x_est, Sigma_est)

    # Store data for plotting.
    x_true_history.append(simulator.x_true())
    x_est_history.append(x_est)
    Sigma_est_history.append(np.diagonal(Sigma_est))

# Convert history lists to arrays.
x_true_history = np.array(x_true_history)
x_est_history = np.array(x_est_history)
Sigma_est_history = np.array(Sigma_est_history)

# Plotting the true path, estimated path, and landmarks.
plt.figure()
plt.plot(x_true_history[:, 0], x_true_history[:, 1], label='True Path')
plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path')
plt.scatter(map.landmarks[:, 0], map.landmarks[:, 1],
            marker='x', color='red', label='Landmarks')
plt.legend()
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('Unicycle Robot Localization using EKF')
plt.axis('equal')
plt.grid(True)
plt.show()

# Note the angle state theta experiences "angles
# wrapping". This small helper function is used
# to address the issue.


# Plot the 2 standard deviation and error history for each state.
state_name = ['x', 'y', 'θ']
estimation_error = x_est_history - x_true_history
estimation_error[:, -1] = wrap_angle(estimation_error[:, -1])
for s in range(3):
    plt.figure()
    two_sigma = 2*np.sqrt(Sigma_est_history[:, s])
    plt.plot(estimation_error[:, s])
    plt.plot(two_sigma, linestyle='dashed', color='red')
    plt.plot(-two_sigma, linestyle='dashed', color='red')
    plt.title(state_name[s])
    plt.show()