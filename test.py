import numpy as np
import torch

from torchsummary import summary

import PPO2

S = 2
T = 2
state_dim_S = (6*6 + 6)
state_dim_T = (3 + 3)
extra = 2

ppo = PPO2.PPO(state_dim_S, state_dim_T, S, T, extra)
ppo.policy_old.load_state_dict(torch.load('model.dat', weights_only=True))

summary(ppo.policy_old, (1, state_dim_S * S + state_dim_T * T + extra))