import matplotlib.pyplot as plt
import numpy as np

data = np.fromfile('ppo_rewards.dat', dtype=np.float32)

plt.plot(data)
plt.show()