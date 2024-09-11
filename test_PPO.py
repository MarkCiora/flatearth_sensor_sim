import numpy as np
import torch
import time

import PPO2
import globals
import sim
from sim import Simulation

import matplotlib.pyplot as plt

floattype = np.float32

S = 2
T = 2
sim_duration = 200

state_dim_S = (3+3)
state_dim_T = (6*6 + 6)
extra = 2
episodes = 200
reps_before_update = 4

ppo = PPO2.PPO(state_dim_S, state_dim_T, S, T, extra)
# ppo.policy.load_state_dict(torch.load('model.dat', weights_only=True)) 
ppo.policy_old.load_state_dict(torch.load('model.dat', weights_only=True))

uncertainty_ppo = np.zeros((int(sim_duration / globals.dt + 1)), dtype=np.float32)
uncertainty_greedy = np.zeros((int(sim_duration / globals.dt + 1)), dtype=np.float32)
for ep in range(episodes):
    print(ep)
    simulation = Simulation(S,T, sim_duration)
    simulation2 = simulation.copy_true()

    # print(simulation.targets, simulation2.targets)

    # for i in range(S):
    #     print(simulation.sensors[i].F, simulation2.sensors[i].F)
    # for i in range(T):
    #     print(simulation.targets[i].x, simulation2.targets[i].x)

    u = [simulation.get_uncertainty()]
    while not(simulation.is_done()):
        x = simulation.get_state_vector()
        action = ppo.select_action_test(x)
        simulation.step(action)
        u.append(simulation.get_uncertainty())
    uncertainty_ppo += np.array(u)

    u = [simulation2.get_uncertainty()]
    while not(simulation2.is_done()):
        slew_times = np.empty((S,T), dtype=floattype)
        gains = np.empty((S,T), dtype=floattype)
        for i in range(S):
            for j in range(T):
                slew_times[i,j] = simulation2.sensors[i].angle_between(simulation2.targets[j].x[0:3]) / simulation.sensors[i].slew_rate
                gains[i,j] = simulation2.targets[j].predict_update_FI(simulation2.sensors[i].p)
        # print(slew_times)
        # slew_times = (slew_times + 0.5) ** m_exponent
        # table = slew_times * gains
        action = np.argmax(gains, axis=1)
        # print(gains, action)
        simulation2.step(action)
        u.append(simulation2.get_uncertainty())
    uncertainty_greedy += np.array(u)

uncertainty_ppo /= episodes
uncertainty_greedy /= episodes

uncertainty_ppo.tofile('uncertainty_ppo.dat')
uncertainty_greedy.tofile('uncertainty_greedy.dat')