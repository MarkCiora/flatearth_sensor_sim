

# BASELINE 1




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
T = 10

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
m_exponent = -1/10
for t in map(lambda x: x * globals.dt, range(int(sim_duration/globals.dt))):
    #CHOOSE TARGETS GREEDY
    slew_times = np.empty((S,T), dtype=floattype)
    traces = np.empty((T), dtype=floattype)
    for i in range(S):
        for j in range(T):
            slew_times[i,j] = sensors[i].angle_between(targets[j].x[0:3]) / sensors[i].slew_rate
    for j in range(T):
        traces[j] = np.trace(targets[j].P)
    slew_times = (slew_times + 1) ** m_exponent
    table = slew_times * traces
    targeting_list = np.argmax(table, axis=1)
    print(targeting_list)

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