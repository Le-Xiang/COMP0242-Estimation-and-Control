import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, dyn_cancel, SinusoidalReference, CartesianDiffKin
from tracker_model import TrackerModel


class LinearReference:
    def __init__(self, slopes, intercepts, initial_angles):
        """
        初始化线性参考生成器。
        
        参数:
        slopes: 每个关节的斜率列表（变化率）
        intercepts: 每个关节的截距列表
        initial_angles: 每个关节的初始角度
        """
        self.slopes = np.array(slopes)
        self.intercepts = np.array(intercepts)
        self.initial_angles = np.array(initial_angles)

    def get_values(self, t):
        """
        在时间 t 获取期望的关节角度和速度。
        
        参数:
        t: 当前时间
        
        返回:
        q_d: 期望的关节位置
        qd_d: 期望的关节速度
        """
        # 线性轨迹: q_d = slope * t + intercept
        q_d = self.slopes * t + self.intercepts
        qd_d = self.slopes  # 斜率是常量，代表速度
        
        return q_d, qd_d
    
class PolynomialReference:
    def __init__(self, coefficients, initial_angles):
        """
        初始化多项式参考生成器。
        
        参数:
        coefficients: 列表的列表，每个内列表包含一个关节的多项式系数（从高到低次幂）
        initial_angles: 每个关节的初始角度
        """
        self.coefficients = [np.array(coeff) for coeff in coefficients]
        self.initial_angles = np.array(initial_angles)

    def get_values(self, t):
        """
        在时间 t 获取期望的关节角度和速度。
        
        参数:
        t: 当前时间
        
        返回:
        q_d: 期望的关节位置
        qd_d: 期望的关节速度
        """
        q_d = []
        qd_d = []
        
        for coeff in self.coefficients:
            # 使用多项式系数计算期望位置
            poly = np.poly1d(coeff)
            q_d.append(poly(t))
            
            # 使用多项式的导数计算期望速度
            poly_derivative = np.polyder(poly)
            qd_d.append(poly_derivative(t))
        
        return np.array(q_d), np.array(qd_d)

def initialize_simulation(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def print_joint_info(sim, dyn_model, controlled_frame_name):
    """Print initial joint angles and limits."""
    init_joint_angles = sim.GetInitMotorAngles()
    init_cartesian_pos, init_R = dyn_model.ComputeFK(init_joint_angles, controlled_frame_name)
    
    print(f"Initial joint angles: {init_joint_angles}")
    
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    print(f"Lower limits: {lower_limits}")
    print(f"Upper limits: {upper_limits}")
    
    joint_vel_limits = sim.GetBotJointsVelLimit()
    print(f"Joint velocity limits: {joint_vel_limits}")
    

def getSystemMatricesContinuos(num_joints, damping_coefficients=None):
    """
    Get the system matrices A and B according to the dimensions of the state and control input.
    
    Parameters:
    sim: Simulation object
    num_joints: Number of robot joints
    damping_coefficients: List or numpy array of damping coefficients for each joint (optional)
    
    Returns:
    A: State transition matrix
    B: Control input matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    
    # Initialize A matrix
    A = np.zeros((num_states,num_states))
    
    # Upper right quadrant of A (position affected by velocity)
    A[:num_joints, num_joints:] = np.eye(num_joints) 
    
    # Lower right quadrant of A (velocity affected by damping)
    #if damping_coefficients is not None:
    #    damping_matrix = np.diag(damping_coefficients)
    #    A[num_joints:, num_joints:] = np.eye(num_joints) - time_step * damping_matrix
    
    # Initialize B matrix
    B = np.zeros((num_states, num_controls))
    
    # Lower half of B (control input affects velocity)
    B[num_joints:, :] = np.eye(num_controls) 
    
    return A, B

# Example usage:
# sim = YourSimulationObject()
# num_joints = 6  # Example: 6-DOF robot
# damping_coefficients = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]  # Example damping coefficients
# A, B = getSystemMatrices(sim, num_joints, damping_coefficients)


def getCostMatrices(num_joints):
    """
    Get the cost matrices Q and R for the MPC controller.
    
    Returns:
    Q: State cost matrix
    R: Control input cost matrix
    """
    num_states = 2 * num_joints
    num_controls = num_joints
    
    # Q = 1 * np.eye(num_states)  # State cost matrix
    p_w = 100000
    v_w = 10
    Q_diag = np.array([p_w, p_w, p_w,p_w, p_w, p_w,p_w, v_w, v_w, v_w,v_w, v_w, v_w,v_w])
    Q = np.diag(Q_diag)
    
    print(Q)

    R = 0.1 * np.eye(num_controls)  # Control input cost matrix
    
    return Q, R


def main():
    # 配置
    conf_file_name = "pandaconfig.json"
    controlled_frame_name = "panda_link8"
    
    # 初始化仿真和动态模型
    sim, dyn_model, num_joints = initialize_simulation(conf_file_name)
    cmd = MotorCommands()
    
    # 打印关节信息
    print_joint_info(sim, dyn_model, controlled_frame_name)
    
    # 初始化数据存储
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []
    regressor_all = np.array([])

    # 定义矩阵
    A, B = getSystemMatricesContinuos(num_joints)
    Q, R = getCostMatrices(num_joints)
    
    # 测量所有状态
    num_states = 2 * num_joints
    C = np.eye(num_states)
    
    # 预测控制的时间长度
    N_mpc = 10

    # 初始化调节器模型
    tracker = TrackerModel(A, B, C, Q, R, N_mpc, num_states, num_joints, num_states, sim.GetTimeStep())
    # 计算预测控制优化需要的矩阵
    S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar = tracker.propagation_model_tracker_fixed_std()
    H,Ftra = tracker.tracker_std(S_bar, T_bar, Q_hat, Q_bar, R_bar)
    
    # 新的参考轨迹
    # 线性参考示例
    slopes = [0.1, 0.2, 0.1, 0.05, 0.05, 0.1, 0.2]  # 每个关节的示例斜率
    intercepts = sim.GetInitMotorAngles()  # 从初始角度开始
    # ref = LinearReference(slopes, intercepts, sim.GetInitMotorAngles())

    # 多项式参考示例
    coefficients = [
        [0.01, -0.1, 0],   # 关节1: 0.01*t^2 - 0.1*t + 0
        [0.02, -0.15, 0],  # 关节2: 0.02*t^2 - 0.15*t + 0
        [0.01, 0, 0],      # 关节3: 0.01*t^2 + 0*t + 0
        [0.005, 0, 0],     # 关节4: 0.005*t^2 + 0*t + 0
        [0.01, -0.05, 0],  # 关节5: 0.01*t^2 - 0.05*t + 0
        [0.02, 0, 0],      # 关节6: 0.02*t^2 + 0*t + 0
        [0.03, -0.1, 0]    # 关节7: 0.03*t^2 - 0.1*t + 0
    ]
    ref = PolynomialReference(coefficients, sim.GetInitMotorAngles())

    # 主控制循环
    episode_duration = 5  # 持续时间（秒）
    current_time = 0
    time_step = sim.GetTimeStep()
    steps = int(episode_duration / time_step)
    sim.ResetPose()
    u_mpc = np.zeros(num_joints)
    for i in range(steps):
        # 测量当前状态
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)

        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        x_ref = []
        # 生成N步的预测轨迹
        for j in range(N_mpc):
            q_d, qd_d = ref.get_values(current_time + j * time_step)
            
            # 叠加 q_d 和 qd_d
            x_ref.append(np.vstack((q_d.reshape(-1, 1), qd_d.reshape(-1, 1))))
        
        x_ref = np.vstack(x_ref).flatten()
        
        # 计算最优控制序列
        u_star = tracker.computesolution(x_ref, x0_mpc, u_mpc, H, Ftra)
        u_mpc += u_star[:num_joints]
       
        # 控制命令
        cmd.tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        sim.Step(cmd, "torque")  # 使用扭矩命令进行仿真步

        # 退出逻辑（按下 'q' 键）
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        
        # 存储用于绘图的数据
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)

        q_d, qd_d = ref.get_values(current_time)

        q_d_all.append(q_d)
        qd_d_all.append(qd_d)

        current_time += time_step
    
    # Plotting
    for i in range(num_joints):
        plt.figure(figsize=(10, 8))
        
        # Position plot for joint i
        plt.subplot(2, 1, 1)
        plt.plot([q[i] for q in q_mes_all], label=f'Measured Position - Joint {i+1}')
        plt.plot([q[i] for q in q_d_all], label=f'Desired Position - Joint {i+1}', linestyle='--')
        plt.title(f'Position Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Position')
        plt.legend()

        # Velocity plot for joint i
        plt.subplot(2, 1, 2)
        plt.plot([qd[i] for qd in qd_mes_all], label=f'Measured Velocity - Joint {i+1}')
        plt.plot([qd[i] for qd in qd_d_all], label=f'Desired Velocity - Joint {i+1}', linestyle='--')
        plt.title(f'Velocity Tracking for Joint {i+1}')
        plt.xlabel('Time steps')
        plt.ylabel('Velocity')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
     
    
    
if __name__ == '__main__':
    
    main()