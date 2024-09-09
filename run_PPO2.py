import numpy as np
import torch
import time

import PPO2
import globals
import sim
from sim import Simulation

import matplotlib.pyplot as plt

S = 2
T = 2
sim_duration = 60

state_dim_S = (6*6 + 6)
state_dim_T = (3 + 3)
extra = 2
episodes = 30000
reps_before_update = 5

ppo = PPO2.PPO(state_dim_S, state_dim_T, S, T, extra)

print("State dim:", state_dim_S, state_dim_T)
print("Sat:", S, "\tTarget:", T)
print(globals.device)

start_time = time.time()

cum_rewards = np.zeros((episodes)).astype(np.float32)
cum_losses = np.zeros((episodes)).astype(np.float32)
for episode in range(episodes):
    for reps in range(reps_before_update):
        simulation = Simulation(S,T, sim_duration)
        while not(simulation.is_done()):
            x = simulation.get_state_vector()
            # print(x.shape)
            action = ppo.select_action(x)
            reward = simulation.step(action) * 1e-1
            # print(reward)
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(simulation.is_done())
            cum_rewards[episode] += reward
    cum_losses[episode] = ppo.update()
    print("Episode", episode, "\treward:", cum_rewards[episode], "\tavg loss:", cum_losses[episode], "\ttime", time.time() - start_time)

total_time = time.time() - start_time
print("total time:", total_time)
cum_rewards.tofile('ppo_rewards.dat')
cum_losses.tofile('cum_losses.dat')
torch.save(ppo.policy.state_dict(), 'model.dat')