import matplotlib.pyplot as plt
import numpy as np

# Load data from files
folder = "."

observations = np.load(f"{folder}/observations_list.npy")
proc_actions = np.load(f"{folder}/proc_actions_list.npy")
raw_actions = np.load("raw_actions_list.npy")
q_log = np.load("q_log.npy")
qdot_log = np.load("qdot_log.npy")

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

# Plot the joint ref - true values
for i in range(len(motor_joint_idxs)):
    plt.figure(figsize=(10, 5))
    plt.plot(q_log[:, :, motor_joint_idxs[i]], label="true")
    plt.plot(proc_actions[:, :, i], label="ref")
    plt.title(motor_joint_names[i])
    plt.xlabel("steps")
    plt.ylabel("position [m]")
    plt.legend()
    plt.grid()
    plt.show()

# Plot the motor joint positions
for i in range(len(motor_joint_idxs)):
    plt.figure(figsize=(10, 5))
    plt.plot(q_log[:, :, motor_joint_idxs[i]], label="q")
    plt.plot(observations[:, :, 12 + i], label="obs")
    plt.title(motor_joint_names[i])
    plt.xlabel("steps")
    plt.ylabel("position [m]")
    plt.legend()
    plt.grid()
    plt.show()

# # Plot the qdot
# for i in range(12):
#     plt.figure(figsize=(10, 5))
#     plt.plot(qdot_log[:, :, motor_joint_idxs[i]], label="qdot", linestyle='-')
#     plt.title(motor_joint_names[i])
#     plt.xlabel("steps")
#     plt.ylabel("velocity [m/s]")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Plot the measured joint positions
# for i in range(len(meas_joint_idxs)):
#     plt.figure(figsize=(10, 5))
#     plt.plot(q_log[:, :, meas_joint_idxs[i]], label="q")
#     plt.plot(observations[:, :, 36 + i], label="obs")
#     plt.title(meas_joint_names[i])
#     plt.xlabel("steps")
#     plt.ylabel("position [m]")
#     plt.legend()
#     plt.grid()
#     plt.show()

# # Plot the raw actions
# for i in range(12):
#     plt.figure(figsize=(10, 5))
#     plt.plot(raw_actions[:, :, i], label="raw")
#     plt.plot(proc_actions[:, :, i], label="proc")
#     plt.plot(observations[:, :, -12 + i], label="past act in obs")
#     plt.title(f"Action {i}")
#     plt.xlabel("steps")
#     plt.ylabel("position [m]")
#     plt.legend()
#     plt.grid()
#     plt.show()


left_current_air_time = np.load(f"{folder}/left_current_air_time.npy")
left_last_air_time = np.load(f"{folder}/left_last_air_time.npy")
right_current_air_time = np.load(f"{folder}/right_current_air_time.npy")
right_last_air_time = np.load(f"{folder}/right_last_air_time.npy")
left_current_contact_time = np.load(f"{folder}/left_current_contact_time.npy")
left_last_contact_time = np.load(f"{folder}/left_last_contact_time.npy")
right_current_contact_time = np.load(f"{folder}/right_current_contact_time.npy")
right_last_contact_time = np.load(f"{folder}/right_last_contact_time.npy")

base_height = np.load(f"{folder}/base_height.npy")
left_step_height = np.load(f"{folder}/left_step_height.npy")
right_step_height = np.load(f"{folder}/right_step_height.npy")
left_contact_forces = np.load(f"{folder}/left_contact_forces.npy")
right_contact_forces = np.load(f"{folder}/right_contact_forces.npy")

# Print statistics for all quantities
quantities = {
    "Left Last Air Time": left_last_air_time,
    "Right Last Air Time": right_last_air_time,
    "Left Last Contact Time": left_last_contact_time,
    "Right Last Contact Time": right_last_contact_time,
    "Base Height": base_height,
    "Left Step Height": left_step_height,
    "Right Step Height": right_step_height,
    "Left Contact Forces": left_contact_forces,
    "Right Contact Forces": right_contact_forces,
}

for name, data in quantities.items():
    print(f"{name}:")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std: {np.std(data):.4f}")
    print(f"  Min: {np.min(data):.4f}")
    print(f"  Max: {np.max(data):.4f}")
    print()


# Prepare data for time plots
all_times = [
    left_current_air_time,
    left_last_air_time,
    right_current_air_time,
    right_last_air_time,
    left_current_contact_time,
    left_last_contact_time,
    right_current_contact_time,
    right_last_contact_time,
]

max_time = max([np.max(times) for times in all_times])
num_steps = len(left_current_air_time)

# Plot base height
plt.figure(figsize=(15, 5))
plt.plot(base_height, label="Base Height", color="blue")
plt.title("Base height")
plt.xlabel("Steps")
plt.ylabel("Base heigh")
plt.grid(True)
plt.legend()
plt.xlim([0, num_steps])
plt.show()

# Plot step heights
plt.figure(figsize=(15, 5))
plt.plot(left_step_height, label="Left Step Height", color="blue")
plt.plot(right_step_height, label="Right Step Height", color="red")
plt.title("Step Heights")
plt.xlabel("Steps")
plt.ylabel("Height")
plt.grid(True)
plt.legend()
plt.xlim([0, num_steps])
plt.show()


# Plot contact forces
plt.figure(figsize=(15, 5))
plt.plot(left_contact_forces, label="Left", color="blue")
plt.plot(right_contact_forces, label="Right", color="red")
plt.title("Contact forces")
plt.xlabel("Steps")
plt.ylabel("f_z [N]")
plt.grid(True)
plt.legend()
plt.xlim([0, num_steps])
plt.show()


# Plot air and contact times comparison
fig, axs = plt.subplots(2, 2, figsize=(15, 10))


def plot_comparison(ax, current_times, last_times, title):
    ax.plot(current_times, label="Current")
    ax.plot(last_times, label="Last")
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Time (s)")
    ax.grid(True)
    ax.legend()
    ax.set_xlim([0, num_steps])
    ax.set_ylim([0, 1.05 * max_time])


plot_comparison(axs[0, 0], left_current_air_time, left_last_air_time, "Left Air Time")
plot_comparison(
    axs[0, 1], right_current_air_time, right_last_air_time, "Right Air Time"
)
plot_comparison(
    axs[1, 0], left_current_contact_time, left_last_contact_time, "Left Contact Time"
)
plot_comparison(
    axs[1, 1], right_current_contact_time, right_last_contact_time, "Right Contact Time"
)

plt.tight_layout()
plt.show()

# Plot current air and contact times
fig, axs = plt.subplots(2, 1, figsize=(15, 10))


def plot_current_times(ax, air_times, contact_times, title):
    ax.plot(air_times, label="Air Time")
    ax.plot(contact_times, label="Contact Time")
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Time (s)")
    ax.grid(True)
    ax.legend()
    ax.set_xlim([0, num_steps])
    ax.set_ylim([0, max_time])


plot_current_times(
    axs[0], left_current_air_time, left_current_contact_time, "Left Current Times"
)
plot_current_times(
    axs[1], right_current_air_time, right_current_contact_time, "Right Current Times"
)

plt.tight_layout()
plt.show()
