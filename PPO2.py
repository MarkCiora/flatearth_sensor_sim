import numpy as np
import torch
from torch import nn
from torch.nn.functional import relu

import globals

device = globals.device

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class Network(nn.Module):
    def __init__(self, state_dim_S, state_dim_T, S, T, extra):
        super(Network, self).__init__()
        self.state_dim_S = state_dim_S
        self.state_dim_T = state_dim_T
        self.S = S
        self.T = T
        self.extra = extra

        self.target_embedding_width = 128
        self.sensor_embedding_width = 128

        self.target_fc1 = nn.Linear(state_dim_T + extra, 256)
        self.target_fc2 = nn.Linear(256, 1024)
        self.target_fc3 = nn.Linear(1024, 512)
        self.target_fc4 = nn.Linear(512, self.target_embedding_width)

        self.sensor_fc1 = nn.Linear(state_dim_S + extra, 256)
        self.sensor_fc2 = nn.Linear(256, 1024)
        self.sensor_fc3 = nn.Linear(1024, 512)
        self.sensor_fc4 = nn.Linear(512, self.sensor_embedding_width)

        self.shared_fc1 = nn.Linear(self.sensor_embedding_width * self.S + \
                                    self.target_embedding_width * self.T, 2048)
        self.shared_fc2 = nn.Linear(2048, 2048)
        self.shared_fc3 = nn.Linear(2048, 2048)
        
        self.actor_fc1 = nn.Linear(2048, 2048)
        self.actor_fc2 = nn.Linear(2048, 1024)
        self.actor_fc3 = nn.Linear(1024, S*T)

        self.value_fc1 = nn.Linear(2048, 1024)
        self.value_fc2 = nn.Linear(1024, 512)
        self.value_fc3 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = x.view(-1, self.state_dim_S * self.S + self.state_dim_T * self.T + self.extra)
        scaling = x[:,self.state_dim_S * self.S + self.state_dim_T * self.T:self.state_dim_S * self.S + self.state_dim_T * self.T + self.extra]
        sensor_input = x[:, 0:self.state_dim_S * self.S]
        target_input = x[:, self.state_dim_S * self.S:self.state_dim_S * self.S + self.state_dim_T * self.T]

        scaling = scaling.unsqueeze(1)
        sensor_input = sensor_input.view(-1, self.S, self.state_dim_S)
        target_input = target_input.view(-1, self.T, self.state_dim_T)

        scaling_sensor = scaling.repeat(1,self.S,1)
        scaling_target = scaling.repeat(1,self.T,1)
        # print(scaling)

        # print(scaling.shape, sensor_input.shape, target_input.shape)

        sensor_input = torch.cat((sensor_input, scaling_sensor), dim=2)
        target_input = torch.cat((target_input, scaling_target), dim=2)
        # print(scaling, sensor_input, target_input)

        x1 = relu(self.target_fc1(target_input))
        x1 = relu(self.target_fc2(x1))
        x1 = relu(self.target_fc3(x1))
        x1 = relu(self.target_fc4(x1))

        x2 = relu(self.sensor_fc1(sensor_input))
        x2 = relu(self.sensor_fc2(x2))
        x2 = relu(self.sensor_fc3(x2))
        x2 = relu(self.sensor_fc4(x2))

        x1 = x1.view(-1, self.T * self.target_embedding_width)
        x2 = x2.view(-1, self.S * self.sensor_embedding_width)

        x = torch.cat((x1, x2), dim=1)
        x = relu(self.shared_fc1(x))
        x = relu(self.shared_fc2(x))
        x = relu(self.shared_fc3(x))
        
        action = relu(self.actor_fc1(x))
        action = relu(self.actor_fc2(action))
        action = self.actor_fc3(action).view(-1, self.S, self.T)

        value = relu(self.value_fc1(x))
        value = relu(self.value_fc2(value))
        value = self.value_fc3(value).view(-1, 1)

        return action, value
    
    def act(self, state):
        action_logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        return action.detach(), action_logprobs.detach(), value.detach()
    
    def act_test(self, state):
        action_logits, value = self.forward(state)
        return torch.argmax(action_logits, dim=1).detach()

    def evaluate(self, state, action):
        action_logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_logprobs, value, entropy
    
class PPO:
    def __init__(self, state_dim_S, state_dim_T, S, T, extra):
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.lr = 5e-5
        self.epochs = 32
        self.S = S
        self.T = T
        self.buffer = RolloutBuffer()
        self.policy = Network(state_dim_S, state_dim_T, S, T, extra).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = Network(state_dim_S, state_dim_T, S, T, extra).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MSELoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logits, value = self.policy_old.act(state)
        # print(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logits)
        self.buffer.state_values.append(value)
        return action.detach().cpu().numpy().squeeze()
    
    def select_action_test(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action = self.policy_old.act_test(state)
        return action.detach().cpu().numpy().squeeze()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = (rewards.detach() - old_state_values.detach()).unsqueeze(1)

        # Optimize policy for K epochs
        cum_loss = 0
        for _ in range(self.epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            # print(logprobs.shape, old_logprobs.shape)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss1 = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            loss2 = 0.5 * self.MSELoss(state_values, rewards)
            
            # take gradient step
            self.optimizer.zero_grad()
            (loss1 + loss2).mean().backward()
            self.optimizer.step()

            cum_loss += loss2
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return cum_loss.detach().mean().item()
    
if __name__ == "__main__":
    state_dim = 20
    S=2
    T=3
    x = np.random.uniform(-1, 1, size=(state_dim)).astype(np.float32)
    # x = torch.from_numpy(x).to(device)
    ppo = PPO(state_dim, S, T)
    action = ppo.select_action(x)
    print(action.dtype)
    # print(net.evaluate(x, action))