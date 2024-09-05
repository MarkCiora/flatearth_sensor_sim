import numpy as np
import matplotlib.pyplot as plt


T = 2

a = np.random.uniform(0,1, size=(10,5,3))
b = a.sum(axis=1)

print(a.shape, b.shape)

