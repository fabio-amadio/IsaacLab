import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array from file
q_log = np.load("q_log.npy")
a_log = np.load("a_log.npy")

env = 0

q_log = q_log[:, env, :]
a_log = a_log[:, env, :]

action_idxs = range(12)
joint_idxs = [4, 11, 18, 19, 36, 37, 38, 40, 43, 45, 46, 48]
joint_names = [
    "leg_left_1_motor",
    "leg_right_1_motor",
    "leg_left_2_motor",
    "leg_left_3_motor",
    "leg_right_2_motor",
    "leg_right_3_motor",
    "leg_left_4_motor",
    "leg_left_5_motor",
    "leg_left_length_motor",
    "leg_right_length_motor",
    "leg_right_4_motor",
    "leg_right_5_motor",
]

# Plot the data
for i in action_idxs:
    plt.figure(figsize=(10, 5))
    plt.plot(q_log[:, joint_idxs[i]], label="true")
    plt.plot(a_log[:, i], label="ref")
    plt.title(joint_names[i])
    plt.xlabel("steps")
    plt.ylabel("position [m]")
    plt.legend()
    plt.grid()
    plt.show()
