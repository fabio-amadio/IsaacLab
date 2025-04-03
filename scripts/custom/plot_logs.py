import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array from file
q_log = np.load("q_log.npy")
a_log = np.load("a_log.npy")


env = 0

q_log = q_log[:, env, :]
a_log = a_log[:, env, :]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(q_log[:, 48], label="Joint")
plt.plot(a_log[:, 11], label="Action")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()


# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(q_log, label="Joint")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

