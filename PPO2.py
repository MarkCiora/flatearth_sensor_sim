import numpy as np
import torch
from torch import nn

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
    def __init__(self, state_dim, S, T):
        super(Network, self).__init__()
        self.state_dim = state_dim
        self.S = S
        self.T = T
        self.fc1a = nn.Linear(state_dim, 1024)
        self.fc2a = nn.Linear(1024, 2048)
        self.fc3a = nn.Linear(2048, 1024)
        self.fc4a = nn.Linear(1024, 512)
        self.fc5a = nn.Linear(512, S*T)
        self.fc1v = nn.Linear(state_dim, 1024)
        self.fc2v = nn.Linear(1024, 2048)
        self.fc3v = nn.Linear(2048, 1024)
        self.fc4v = nn.Linear(1024, 512)
        self.fc5v = nn.Linear(512, 1)
        
    def forward(self, x):
        x1 = nn.ReLU(self.fc1a(x))
        x1 = nn.ReLU(self.fc2a(x1))
        x1 = nn.ReLU(self.fc3a(x1))
        x1 = nn.ReLU(self.fc4a(x1))
        action = self.fc5a(x1)
        x2 = nn.ReLU(self.fc1v(x))
        x2 = nn.ReLU(self.fc2v(x2))
        x2 = nn.ReLU(self.fc3v(x2))
        x2 = nn.ReLU(self.fc4v(x2))
        value = self.fc5v(x2)
        return action, value
    
    def act(self, state):
        # action_logits = self.actor(state).view(-1, self.S, self.T)
        # value = self.critic(state)
        action_logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        return action.detach(), action_logprobs.detach(), value.detach()
    
    def evaluate(self, state, action):
        # action_logits = self.actor(state).view(-1, self.S, self.T)
        # value = self.critic(state)
        action_logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_logprobs, value, entropy
    
class PPO:
    def __init__(self, state_dim, S, T):
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.lr_actor = 3e-5
        self.lr_critic = 1e-5
        self.epochs = 100
        self.state_dim = state_dim
        self.S = S
        self.T = T
        self.buffer = RolloutBuffer()
        self.policy = Network(state_dim, S, T).to(device)
        self.optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
            ])
        self.policy_old = Network(state_dim, S, T).to(device)
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