import numpy as np
import matplotlib.pyplot as plt

N = 10
iterations = 200

points = np.random.uniform(0.0, 10.0, (N,2))
path = np.random.permutation(N)
original_path = path.copy()

selections = np.random.randint(0,N,iterations)

def score_path(path_):
    v = 0
    for i in range(path_.shape[0]):
        v += np.linalg.norm(path_[i-1,:] - path_[i,:])
    return v

# print(score_path(points[original_path,:]))

for i in range(iterations):
    sel = [(selections[i] + j) % N for j in range(4)]
    sel_alt = [(selections[i] + j) % N for j in range(4)]
    sel_alt[1], sel_alt[2] = sel_alt[2], sel_alt[1]

    path_sel = path[sel]
    points_sel = points[path_sel,:]
    path_sel_alt = path[sel_alt]
    points_sel_alt = points[path_sel_alt,:]
    s0 = score_path(points_sel)
    s1 = score_path(points_sel_alt)
    if s1 < s0:
        a = selections[i]
        # print(path, (a+1)%N, (a+2)%N)
        path[(a+1)%N], path[(a+2)%N] = path[(a+2)%N], path[(a+1)%N]

print("Original score:", score_path(points[original_path,:]))
print("New score:", score_path(points[path,:]))

plt.scatter(points[:, 0], points[:, 1], color='green')
plt.plot(points[np.append(path, path[0]),0], points[np.append(path,path[0]),1], color='green', linestyle='-', marker='o')
plt.plot(points[np.append(original_path, original_path[0]),0], points[np.append(original_path,original_path[0]),1], color='blue', linestyle='-', marker='o')
plt.show()

