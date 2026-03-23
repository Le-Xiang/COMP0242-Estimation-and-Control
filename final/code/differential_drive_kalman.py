import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel

# 在文件顶部导入卡尔曼滤波器
from robot_localization_system import RobotEstimator, FilterConfiguration, Map

# global variables
W_range = 0.5 ** 2  # Measurement noise variance (range measurements)
W_bearing = (np.pi * 0.5 / 180.0) ** 2
landmarks = Map().landmarks

def landmark_range_bearing_observations_new(base_position):
    """生成距离-方位测量数据"""
    y = []
    Wr = W_range  # 距离测量噪声
    Wb = W_bearing  # 方位角测量噪声

    for lm in landmarks:
        # 计算地标与机器人之间的相对距离和方位角
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_true = np.sqrt(dx**2 + dy**2)
        bearing_true = np.arctan2(dy, dx) - base_position[2]

        # 添加测量噪声
        range_meas = range_true + np.random.normal(0, np.sqrt(Wr))
        bearing_meas = bearing_true + np.random.normal(0, np.sqrt(Wb))

        # 角度包裹，确保方位角在 [-π, π] 范围内
        bearing_meas = wrap_angle(bearing_meas)

        # 将测量值（距离和方位角）添加到观测列表中
        y.append([range_meas, bearing_meas])

    return np.array(y)

def landmark_range_observations(base_position):
    y = []
    C = []
    W = W_range
    for lm in landmarks:
        # True range measurement (with noise)
        dx = lm[0] - base_position[0]
        dy = lm[1] - base_position[1]
        range_meas = np.sqrt(dx**2 + dy**2)
       
        y.append(range_meas)

    y = np.array(y)
    return y


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


def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints

# 在初始化时创建卡尔曼滤波器实例
def init_simulator_with_kf(conf_file_name):
    sim, dyn_model, num_joints = init_simulator(conf_file_name)
    
    # 初始化卡尔曼滤波器
    filter_config = FilterConfiguration()  # 根据需要配置
    map_data = Map()  # 初始化地图数据
    kf_estimator = RobotEstimator(filter_config, map_data)
    kf_estimator.start()  # 启动卡尔曼滤波器
    
    return sim, dyn_model, num_joints, kf_estimator



def main():
    # Configuration for the simulation
    conf_file_name = "robotnik.json"  # Configuration file for the robot
    # sim,dyn_model,num_joints=init_simulator(conf_file_name)
    #其实也是用来mpc simulator的部分的
    sim, dyn_model, num_joints, kf_estimator = init_simulator_with_kf(conf_file_name)
    
    # print("--------------------------------------------------")
    # print(f"num_joints is :{num_joints}")

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
    init_pos  = np.array([2.0, 3.0])
    init_quat = np.array([0,0,0.3827,0.9239]) #机器人在 z 轴上旋转了 45 度的角度
    init_base_bearing_ = quaternion2bearing(init_quat[3], init_quat[0], init_quat[1], init_quat[2]) #将四元数转换为偏航角的函数
    cur_state_x_for_linearization = [init_pos[0], init_pos[1], init_base_bearing_] #定义了用于系统线性化的当前状态（初始时刻）[2.0,3.0,0.7854]
    cur_u_for_linearization = np.zeros(num_controls) #初始时刻没有控制输入（即机器人没有移动或旋转）
    regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)
    # Define the cost matrices
    Qcoeff = np.array([310.0, 310.0, 80.0])
    Rcoeff = 0.5
    regulator.setCostMatrices(Qcoeff,Rcoeff)
    u_mpc = np.zeros(num_controls)
    ##### robot parameters ########
    wheel_radius = 0.11
    wheel_base_width = 0.46
  
    ##### MPC control action #######
    v_linear = 0.0
    v_angular = 0.0


    #kalman 设置初始控制命令和其他参数
    cmd = MotorCommands()  # Initialize command structure for motors
    init_angular_wheels_velocity_cmd = np.array([0.0, 0.0, 0.0, 0.0])
    init_interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
    cmd.SetControlCmd(init_angular_wheels_velocity_cmd, init_interface_all_wheels)

    # Arrays to store data for plotting
    x_true_history = []
    x_est_history = []
    Sigma_est_history = []
    episode_duration = 30
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration/time_step)
    step=0
    # Main control loop
    while True:
        # True state propagation (with process noise)

        
        ##### advance simulation ##################################################################
        time_step = sim.GetTimeStep()

        #TODO Kalman filter prediction
    
        # Get the measurements from the simulator ###########################################
        # measurements of the robot without noise (just for comparison purpose) #############
        base_pos_no_noise = sim.bot[0].base_position
        base_ori_no_noise = sim.bot[0].base_orientation
        base_bearing_no_noise_ = quaternion2bearing(base_ori_no_noise[3], base_ori_no_noise[0], base_ori_no_noise[1], base_ori_no_noise[2])
        base_lin_vel_no_noise  = sim.bot[0].base_lin_vel
        base_ang_vel_no_noise  = sim.bot[0].base_ang_vel

        # Measurements of the current state (real measurements with noise) ##################################################################
        base_pos = sim.GetBasePosition()
        base_ori = sim.GetBaseOrientation()
        base_bearing_ = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        # y = landmark_range_observations(base_pos)
    
        # 从 MPC 获取控制输入
        # cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        # cur_u_for_linearization = u_mpc
        # regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization) #既是更新，也是为了得到一个AB，初始的AB！！

        # TODO Update the filter with the latest observations
        # 卡尔曼滤波器预测步骤（使用了robot_localization_system里面的estimator的类）
        kf_estimator.set_control_input(cur_u_for_linearization)
        kf_estimator.predict_to(current_time)
        # 获取测量数据并更新卡尔曼滤波器
        y = landmark_range_bearing_observations_new(cur_state_x_for_linearization)  # 或 base_pos_all 中的其他观测数据
        kf_estimator.update_from_landmark_range_bearing_observations(y)  # 是卡尔曼滤波器中的更新方法，它会使用 y 来计算创新量、卡尔曼增益，并更新状态和协方差。
    
        # TODO Get the current state estimate
        # 使用更新后的状态作为 MPC 优化的基础
        #x_est 是一个向量，包含当前时刻的状态估计，例如机器人的位置 𝑥y 坐标和角度  
        # Sigma_est 是一个协方差矩阵，用于描述状态估计的不确定性。如果矩阵的值较大，则表示状态的不确定性较高。 
        x_est, Sigma_est = kf_estimator.estimate()

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
        
        # Compute the matrices needed for MPC optimization
        # TODO here you want to update the matrices A and B at each time step if you want to linearize around the current points
        # add this 3 lines if you want to update the A and B matrices at each time step 
        
        cur_state_x_for_linearization = [base_pos[0], base_pos[1], base_bearing_]
        cur_u_for_linearization = u_mpc
        regulator.updateSystemMatrices(sim,cur_state_x_for_linearization,cur_u_for_linearization)

        #最优J的计算部分
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        # Compute the optimal control sequence
        H_inv = np.linalg.inv(H)

        # 将卡尔曼滤波器的状态估计传递给控制器（如 MPC），作为控制器计算下一个控制输入的基础
        u_mpc = -H_inv @ F @ x_est
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls] 

        # Prepare control command to send to the low level controller
        left_wheel_velocity,right_wheel_velocity=velocity_to_wheel_angular_velocity(u_mpc[0],u_mpc[1], wheel_base_width, wheel_radius)
        angular_wheels_velocity_cmd = np.array([right_wheel_velocity, left_wheel_velocity, left_wheel_velocity, right_wheel_velocity])
        interface_all_wheels = ["velocity", "velocity", "velocity", "velocity"]
        cmd.SetControlCmd(angular_wheels_velocity_cmd, interface_all_wheels)

        sim.Step(cmd, "torque")
        print(f"u_mpc:{u_mpc}")
        print(f"position X :{base_pos}")

         # Store data for plotting.
        x_true_history.append(cur_state_x_for_linearization)
        x_est_history.append(x_est)
        Sigma_est_history.append(np.diagonal(Sigma_est))
        
        step=step+1
        if step == steps:
            break

        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # Store data for plotting if necessary
        base_pos_all.append(base_pos)
        base_bearing_all.append(base_bearing_)

        # Update current time
        current_time += time_step
        



    # Plotting the trajectory
    base_pos_all = np.array(base_pos_all)
    # Convert history lists to arrays.
    x_true_history = np.array(x_true_history)
    x_est_history = np.array(x_est_history)
    Sigma_est_history = np.array(Sigma_est_history)

    # Plotting 
    #add visualization of final x, y, trajectory and theta
    plt.figure()
    plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label='Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Robot Trajectory(cost of time:{current_time})')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))

    # 绘制 X 位置
    plt.subplot(3, 1, 1)
    plt.plot(base_pos_all[:, 0], label='X Position')
    plt.ylabel('X Position')
    plt.legend()

    # 绘制 Y 位置
    plt.subplot(3, 1, 2)
    plt.plot(base_pos_all[:, 1], label='Y Position')
    plt.ylabel('Y Position')
    plt.legend()

    # 绘制偏航角 θ
    plt.subplot(3, 1, 3)
    plt.plot(base_bearing_all, label='Theta (Yaw)')
    plt.xlabel('Time Step')
    plt.ylabel('Theta (Yaw)')
    plt.legend()

    # 展示和保存图像
    plt.tight_layout()
    plt.show()

    # Plotting the true path, estimated path, and landmarks.
    plt.figure()
    plt.plot(x_true_history[:, 0], x_true_history[:, 1], label='True Path')
    plt.plot(x_est_history[:, 0], x_est_history[:, 1], label='Estimated Path')
    plt.scatter(landmarks[:, 0], landmarks[:, 1],
                marker='x', color='red', label='Landmarks')
    plt.legend()
    plt.xlabel('X position [m]')
    plt.ylabel('Y position [m]')
    plt.title(f'Robot Localization using EKF (N={N_mpc})')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

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


    

if __name__ == '__main__':
    main()