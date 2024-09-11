import matplotlib.pyplot as plt
import numpy as np

window_width = 50
window = np.zeros((window_width), dtype=np.float32) + 1/window_width

data_ppo = np.fromfile('uncertainty_ppo.dat', dtype=np.float32)
data_greedy = np.fromfile('uncertainty_greedy.dat', dtype=np.float32)

plt.plot(data_ppo, color='blue')
plt.plot(data_greedy, color='red')
plt.show()
