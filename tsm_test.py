import numpy as np
import matplotlib.pyplot as plt


T = 2

a = np.zeros((3,3,T))
b = np.zeros((3,T))

print(a,b)


# print(a.flatten(),b.flatten())

c = np.append(a,b)
print(c)