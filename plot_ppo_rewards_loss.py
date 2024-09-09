import matplotlib.pyplot as plt
import numpy as np

window_width = 50
window = np.zeros((window_width), dtype=np.float32) + 1/window_width

data = np.fromfile('ppo_rewards.dat', dtype=np.float32)
data_ma = np.convolve(data, window)

plt.plot(data, color='blue')
plt.plot(data_ma, color='red')
plt.show()


data = np.fromfile('cum_losses.dat', dtype=np.float32)
data_ma = np.convolve(data, window)

plt.plot(data, color='blue')
plt.plot(data_ma, color='red')
plt.show()