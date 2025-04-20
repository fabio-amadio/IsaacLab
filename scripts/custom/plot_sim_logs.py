import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array from file
q_log = np.load("q_log.npy")
a_log = np.load("a_log.npy")
# a_log = np.load("proc_actions_list.npy")
# observations = np.load("observations_list.npy")

env = 0

q_log = q_log[:, env, :]
a_log = a_log[:, env, :]

action_idxs = range(12)
motor_joint_idxs = [4, 11, 18, 19, 36, 37, 38, 40, 43, 45, 46, 48]
motor_joint_names = [
    "leg_left_1_motor",  # joint_idx: 4
    "leg_right_1_motor",  # joint_idx: 11
    "leg_left_2_motor",  # joint_idx: 18
    "leg_left_3_motor",  # joint_idx: 19
    "leg_right_2_motor",  # joint_idx: 36
    "leg_right_3_motor",  # joint_idx: 37
    "leg_left_4_motor",  # joint_idx: 38
    "leg_left_5_motor",  # joint_idx: 40
    "leg_left_length_motor",  # joint_idx: 43
    "leg_right_length_motor",  # joint_idx: 45
    "leg_right_4_motor",  # joint_idx: 46
    "leg_right_5_motor",  # joint_idx: 48
]


meas_joint_idxs = [1, 2, 7, 8, 14, 15, 21, 23, 30, 32]
meas_joint_names = [
    "leg_left_1_joint",  # joint_idx: 1
    "leg_right_1_joint",  # joint_idx: 2
    "leg_left_2_joint",  # joint_idx: 7
    "leg_right_2_joint",  # joint_idx: 8
    "leg_left_3_joint",  # joint_idx: 14
    "leg_right_3_joint",  # joint_idx: 15
    "left_ankle_4_pendulum_joint",  # joint_idx: 21
    "left_ankle_5_pendulum_joint",  # joint_idx: 23
    "right_ankle_4_pendulum_joint",  # joint_idx: 30
    "right_ankle_5_pendulum_joint",  # joint_idx: 32
]

"""Check actions"""

# Plot the data
for i in action_idxs:
    plt.figure(figsize=(10, 5))
    plt.plot(q_log[:, motor_joint_idxs[i]], label="true")
    plt.plot(a_log[:, i], label="ref")
    plt.title(motor_joint_names[i])
    plt.xlabel("steps")
    plt.ylabel("position [m]")
    plt.legend()
    plt.grid()
    plt.show()


"""Check observations"""
# +---------------------------------------------------------+
# | Active Observation Terms in Group: 'policy' (shape: (68,)) |
# +-----------+---------------------------------+-----------+
# |   Index   | Name                            |   Shape   |
# +-----------+---------------------------------+-----------+
# |     0     | base_lin_vel                    |    (3,)   |
# |     1     | base_ang_vel                    |    (3,)   |
# |     2     | projected_gravity               |    (3,)   |
# |     3     | velocity_commands               |    (3,)   |
# |     4     | motor_joint_pos                 |   (12,)   |
# |     5     | motor_joint_vel                 |   (12,)   |
# |     6     | measured_joint_pos              |   (10,)   |
# |     7     | measured_joint_vel              |   (10,)   |
# |     8     | actions                         |   (12,)   |
# +-----------+---------------------------------+-----------+

# # Plot the motor joint positions
# for i in range(len(motor_joint_idxs)):
#     plt.figure(figsize=(10, 5))
#     plt.plot(q_log[:, motor_joint_idxs[i]], label="q")
#     plt.plot(observations[:, :, 12 + i], label="obs")
#     plt.title(motor_joint_names[i])
#     plt.xlabel("steps")
#     plt.ylabel("position [m]")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Plot the measured joint positions
# for i in range(len(meas_joint_idxs)):
#     plt.figure(figsize=(10, 5))
#     plt.plot(q_log[:, meas_joint_idxs[i]], label="q")
#     plt.plot(observations[:, :, 36 + i], label="obs")
#     plt.title(meas_joint_names[i])
#     plt.xlabel("steps")
#     plt.ylabel("position [m]")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Plot the raw actions
# raw_actions = np.load("raw_actions_list.npy")
# for i in range(12):
#     plt.figure(figsize=(10, 5))
#     plt.plot(raw_actions[:, :, i], label="act")
#     plt.plot(observations[:, :, -12 + i], label="obs")
#     plt.title(f"Raw action {i}")
#     plt.xlabel("steps")
#     plt.ylabel("position [m]")
#     plt.legend()
#     plt.grid()
#     plt.show()
