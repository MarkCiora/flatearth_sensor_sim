import numpy as np
import torch
import time

import PPO
import globals
import sim
from sim import Simulation

import matplotlib.pyplot as plt

S = 2
T = 2
sim_duration = 60

state_dim = (6*6 + 6)*T + (3 + 3)*S + 2
episodes = 100
reps_before_update = 5

ppo = PPO.PPO(state_dim, S, T)
# ppo.policy.load_state_dict(torch.load('model.dat', weights_only=True)) 
ppo.policy_old.load_state_dict(torch.load('model.dat', weights_only=True))

uncertainty = np.zeros((int(sim_duration / globals.dt + 1)), dtype=np.float32)
for ep in range(episodes):
    simulation = Simulation(S,T, sim_duration)
    u = [simulation.get_uncertainty()]
    while not(simulation.is_done()):
        x = simulation.get_state_vector()
        action = ppo.select_action(x)
        simulation.step(action)
        ppo.buffer.is_terminals.append(simulation.is_done())
        u.append(simulation.get_uncertainty())
    uncertainty += np.array(u)
uncertainty /= episodes

print(uncertainty.shape)
uncertainty.tofile('uncertainty.dat')