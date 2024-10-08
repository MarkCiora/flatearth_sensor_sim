
# DOES NOT WORK
# CLEARLY CONVERGES TO VERY POOR RESULT REGARDLESS OF PARAMETERS
# VERY SLOW TO OPTIMIZE, NOT FEASIBLE REAL TIME


import numpy as np
import matplotlib.pyplot as plt

import globals
from globals import floattype
import functions
import sim

#sim bounds

target_height_bound = [10, 400]
target_radius_bound = 1500
target_speed_range = [2,2.5]

sensor_height_bound = [1000, 2000]
sensor_radius_bound = 2500

sim_duration = 100

S = 3
T = 5

sensors = []
targets = [] 

for i in range(S):
    angle = i * 2*np.pi / S
    b1 = np.cos(angle)
    b2 = np.sin(angle)
    sensor = sim.Sensor()
    sensor.p[0] = b1 * sensor_radius_bound
    sensor.p[1] = b2 * sensor_radius_bound
    sensor.p[2] = 1500
    sensors.append(sensor)
    # print(sensor.p)

for i in range(T):
    target = sim.Target()
    height = np.random.uniform(target_height_bound[0], target_height_bound[1], 1)[0]
    pos01 = functions.random2d() * target_radius_bound
    speed = np.random.uniform(target_speed_range[0], target_speed_range[1], 1)
    dir01 = functions.random2d()
    target.x[0:2] = pos01
    target.x[2] = height
    target.x[3:5] = dir01 * speed
    target.x_[:] = np.random.multivariate_normal(target.x[:], target.P)
    target.x_[3:6] *= 0
    targets.append(target)

targeting_list = [0 for i in range(S)]
avg_error = []
H = 100
step_size = 1e-3
steps = 2000

for t in map(lambda x: x * globals.dt, range(int(sim_duration/globals.dt))):
    #CHOOSE TARGETS GREEDY
    print("next time", t)
    probs = np.ones((H,S,T), dtype=floattype) / T
    realization = np.empty((H,S), dtype=np.uint16)
    for step in range(steps):
        for h in range(H):
            for i in range(S):
                realization[h,i] = np.random.choice(T, p=probs[h,i,:])
        Vstart = 0 # want this to shrink
        for j in range(T):
            Vstart += np.sqrt(np.trace(targets[j].P))
        targets_copy = [target.copy() for target in targets]
        sensors_copy = [sensor.copy() for sensor in sensors]
        for h in range(H):
            for target in targets_copy:
                target.propagate()
            for i in range(S):
                if sensors_copy[i].propagate(targets_copy[realization[h,i]].x_[0:3]):
                    for j in range(T):
                        if sensors_copy[i].check_in_fov(targets_copy[j].x[0:3]):
                            targets_copy[j].update(sensors_copy[i].p)
        Vend = 0 # want this to shrink
        for j in range(T):
            Vend += np.sqrt(np.trace(targets_copy[j].P))
        Vchange = Vend - Vstart
        Jgrad = np.zeros((H,S,T), dtype=floattype)
        for h in range(H):
            for i in range(S):
                Jgrad[h,i,realization[h,i]] = 1 / probs[h,i,realization[h,i]]
        Jgrad *= Vchange
        probs = probs + step_size * Jgrad
        probs[probs < 0] = 0
        for h in range(H):
            for i in range(S):
                probs[h,i,:] /= np.sum(probs[h,i,:])
    print(probs[0,:,:])
    for i in range(S):
        targeting_list[i] = np.argmax(probs[0,i,:])

    #PROPAGATE
    for target in targets:
        target.propagate()
    for i in range(S):
        if sensors[i].propagate(targets[targeting_list[i]].x_[0:3]):
            for j in range(T):
                if sensors[i].check_in_fov(targets[j].x[0:3]):
                    targets[j].update(sensors[i].p)
    
    #THIS SHOULD NOT AFFECT SIM
    error = 0
    for target in targets:
        error += np.linalg.norm(target.x[0:3] - target.x_[0:3])
    avg_error.append(error / T)

plt.plot(np.array(avg_error))
plt.show()