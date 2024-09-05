import numpy as np
import torch
import time

import PPO
import globals
import sim
from sim import Simulation

import matplotlib.pyplot as plt

floattype = np.float32

S = 2
T = 2
sim_duration = 60

state_dim = (6*6 + 6)*T + (3 + 3)*S + 2

episodes = 1
m_exponent = -1/10

uncertainty = np.zeros((int(sim_duration / globals.dt + 1)), dtype=np.float32)
for ep in range(episodes):
    simulation = Simulation(S,T, sim_duration)
    u = [simulation.get_uncertainty()]
    while not(simulation.is_done()):
        slew_times = np.empty((S,T), dtype=floattype)
        gains = np.empty((S,T), dtype=floattype)
        for i in range(S):
            for j in range(T):
                slew_times[i,j] = simulation.sensors[i].angle_between(simulation.targets[j].x[0:3]) / simulation.sensors[i].slew_rate
                gains[i,j] = simulation.targets[j].predict_update_FI(simulation.sensors[i].p)
        slew_times = (slew_times + 0.5) ** m_exponent
        table = slew_times * gains
        print(table)
        action = np.argmax(table, axis=1)
        simulation.step(action)
        u.append(simulation.get_uncertainty())
    uncertainty += np.array(u)
uncertainty /= episodes

print(uncertainty.shape)
uncertainty.tofile('uncertainty_greedy.dat')