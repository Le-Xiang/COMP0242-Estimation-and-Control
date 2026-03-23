import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel
import json
from robot_localization_system import RobotEstimator, FilterConfiguration, Map

from test_func import Dataset
import pandas as pd

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
W_bearing = (np.pi * 0.5 / 180.0) ** 2  # Measurement noise variance (bearing measurements)

col = ["init_x_idx", "loop_num", "base_pos_x", "base_pos_y", \
        "base_bearing", "x_true_x", "x_true_y", \
        "x_est_x", "x_est_y", "Sigma_est"]

# new function for EKF
def landmark_range_bearing_observations(base_position, landmarks, W_range, W_bearing):
    y = []
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2) + np.random.normal(0, np.sqrt(W_range))
        bearing_meas = np.arctan2(dy, dx) - base_position[2] + np.random.normal(0, np.sqrt(W_bearing))
        bearing_meas = wrap_angle(bearing_meas)
        y.append([range_meas, bearing_meas])

    y = np.array(y)
    return y

#将四元数转换为方位角（航向角），这是在机器人仿真中使用的姿态表示之一。
def quaternion2bearing(q_w, q_x, q_y, q_z):
    quat = pin.Quaternion(q_w, q_x, q_y, q_z)
    quat.normalize()  # Ensure the quaternion is normalized

    # Convert quaternion to rotation matrix
    rot_quat = quat.toRotationMatrix()

    # Convert rotation matrix to Euler angles (roll, pitch, yaw)
    base_euler = pin.rpy.matrixToRpy(rot_quat)  # Returns [roll, pitch, yaw]

    # Extract the yaw angle
    bearing_ = base_euler[2]

    return bearing_

#初始化仿真器和动力学模型，读取配置文件 robotnik.json 以配置机器人和仿真环境
def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=True)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints

# read config from the json file.
def read_init_config(conf_file_name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
        # reading json file and instatiate some variables
    
    if cur_dir:
        conf_file_path = os.path.join(cur_dir,'configs',conf_file_name)
    else:
        conf_file_path = os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,'configs',conf_file_name)
    with open(conf_file_path) as json_file:
            config = json.load(json_file)

    return config

def init_ekf(conf_file_name, x0):
    sim, dyn_model, num_joints = init_simulator(conf_file_name)
    filter_config = FilterConfiguration()
    map_data = Map()

    # get the configuration from the json file
    config = read_init_config(conf_file_name)
    pos_noise = config.get("robot_pybullet").get("robot_noise")[0].get("base_pos_cov")
    ori_noise = config.get("robot_pybullet").get("robot_noise")[0].get("base_ori_cov")
    # change some parameters in FilterConfiguration
    filter_config.Sigma0 =  np.diag([pos_noise, pos_noise, ori_noise]) ** 2
    filter_config.x0 = x0

    # get the initial position and orientation of the robot
    kf_estimator = RobotEstimator(filter_config, map_data)
    kf_estimator.start()  # 启动卡尔曼滤波器

    return sim, dyn_model, num_joints, kf_estimator, config, map_data


def main(init_x_idx, x0, loop_num, df):
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    # Initialize the simulator and the dynamic model
    sim, dyn_model, num_joints, kf_estimator, config, map = init_ekf(conf_file_name, x0)

    # adjusting floor friction
    floor_friction = 100
    sim.SetFloorFriction(floor_friction)
    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    # Initialize data storage
    base_pos_all, base_bearing_all = [], []

    # initializing MPC
    # Define the matrices
    num_states = 3
    num_controls = 2
   
    # Measuring all the state
    C = np.eye(num_states)
    
    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states) # N, q, m, n  
    # update A,B,C matrices
    # TODO provide state_x_for_linearization,cur_u_for_linearization to linearize the system
    # you can linearize around the final state and control of the robot (everything zero)
    # or you can linearize around the current state and control of the robot
    # in the second case case you need to update the matrices A and B at each time step
    # and recall everytime the method updateSystemMatrices

    # config_pos = np.array(config.get("robot_pybullet").get("init_link_base_position")[0])
    # config_quat = np.array(config.get("robot_pybullet").get("init_link_base_orientation")[0])
    # init_base_bearing_ = quaternion2bearing(config_quat[3], config_quat[0], config_quat[1], config_quat[2])
    # # init_pos = np.array([config_pos[0], config_pos[1]])
    # cur_state_x_for_linearization = [config_pos[0], config_pos[1], init_base_bearing_] #定义了用于系统线性化的当前状态（初始时刻）[2.0,3.0,0.7854]
    cur_state_x_for_linearization = [x0[0], x0[1], x0[2]]
    cur_u_for_linearization = np.zeros(num_controls) #初始时刻没有控制输入（即机器人没有移动或旋转）
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    print(f"cur_state_x_for_linearization:{cur_state_x_for_linearization}")
    
    # Define the cost matrices
    Qcoeff = np.array([310.0, 500.0, 230.0])
    Rcoeff = [0.8, 0.1]
    regulator.setCostMatrices(Qcoeff,Rcoeff)
    u_mpc = np.zeros(num_controls)
    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    v_linear = 0.0
    v_angular = 0.0

    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    # store history
    x_true_history = []
    x_est_history = []
    Sigma_est_history = []
    episode_duration = 15
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    step = 0
    
    #TODO 选择一个合适的N
    # Main control loop
    while True:
        # True state propagation (with process noise)
        ##### advance simulation ##################################################################
        time_step = sim.GetTimeStep()

        #TODO Kalman filter prediction
        kf_estimator.set_control_input(cur_u_for_linearization)
        kf_estimator.predict_to(current_time)

        # Get the measurements from the simulator ###########################################
        # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel
        base_pos_ori_no_noise = [base_pos_no_noise[0], base_pos_no_noise[1], base_bearing_no_noise_]
        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        # y = landmark_range_observations(base_pos)
        y = landmark_range_bearing_observations(base_pos_ori_no_noise,map.landmarks,W_range,W_bearing)
    
        # TODO Update the filter with the latest observations
        kf_estimator.update_with_range_bearing(y)

        # TODO Get the current state estimate
        x_est, Sigma_est = kf_estimator.estimate()


        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
        
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        # add this 3 lines if you want to update the A and B matrices at each time step 
        cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        regulator.updateSystemMatrices(sim,x_est,cur_u_for_linearization)

        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)
        u_mpc = -H_inv @ F @ x_est
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls]

        cur_u_for_linearization = u_mpc


        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        sim.Step(cmd, "torque")
        # print(f"u_mpc:{u_mpc}")
        # print(f"base_pos:{base_pos}")

        # # Store data for plotting
        # x_true_history.append(cur_state_x_for_linearization)
        # x_est_history.append(x_est)
        # Sigma_est_history.append(np.diagonal(Sigma_est))
        # # if np.linalg.norm(u_mpc) < 1.1:  # 当控制输入非常小时停止
        # #     break

        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        if step >= steps:
            break
    

        # Store data for plotting if necessary
        base_pos_all.append(base_pos_no_noise)
        base_bearing_all.append(base_bearing_no_noise_)
        # x_true_history.append(cur_state_x_for_linearization)
        x_true_history.append(base_pos_ori_no_noise)
        x_est_history.append(x_est)
        Sigma_est_history.append(np.diagonal(Sigma_est))

        # Update current time
        current_time += time_step
        step += 1

    # save the list to the dataframe
    idx_list = [init_x_idx] * len(x_true_history)
    loop_num_list = [loop_num] * len(x_true_history)
    base_pos_x = [x[0] for x in base_pos_all]
    base_pos_y = [x[1] for x in base_pos_all]
    x_true_x = [x[0] for x in x_true_history]
    x_true_y = [x[1] for x in x_true_history]
    x_est_x = [x[0] for x in x_est_history]
    x_est_y = [x[1] for x in x_est_history]
    Sigma_est = Sigma_est_history

    new_data = pd.DataFrame(list(zip(idx_list, loop_num_list, base_pos_x, base_pos_y,\
                                    base_bearing_all, x_true_x, x_true_y, \
                                        x_est_x, x_est_y, Sigma_est)), columns=col)
                                 

    df = pd.concat([df, new_data], ignore_index=True)
    print(f"df shape: {df.shape}")
    
    return
    


if __name__ == '__main__':
    # get the dataset
    experiment_type_list = ["position", "angle", "distance"]
    dataset = Dataset()

    for experiment_type in experiment_type_list:
        dataset.set_experiment_type(experiment_type)
        initial_states = dataset.initial_states_by_experiment_type()
        print(f"Initial states for the robot for {experiment_type} experiment:")
        print(initial_states)

        # initial the dataframe
        df = pd.DataFrame(columns=col)
        for j in range(len(initial_states)):
            x0 = initial_states[j]
            print(f"Initial state: {x0}")
            
            # run the simulation for n times
            for i in range(dataset.num_experiments):
                print(f"Experiment {i+1}")
                main(j, x0, i+1, df)
                
            
        # save the dataframe to csv file
        df.to_csv(f"results/results_{experiment_type}_{j}.csv", index=False)