import matplotlib.pyplot as plt
import numpy as np

data = np.fromfile('uncertainty_greedy.dat', dtype=np.float32)

plt.plot(data)
plt.show()