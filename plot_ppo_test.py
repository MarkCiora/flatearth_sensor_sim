import matplotlib.pyplot as plt
import numpy as np

data = np.fromfile('uncertainty.dat', dtype=np.float32)

plt.plot(data)
plt.show()