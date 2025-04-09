import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# Hyperparameters
LR = 0.0003
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_EPOCHS = 10
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.001


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.feature_net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())

        # Actor (policy network)
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim)
        )

        # Log standard deviation (learned parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Critic (value network)
        self.critic = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

        self.max_action = max_action

    def forward(self, state):
        features = self.feature_net(state)

        # Actor: mean and log_std of action distribution
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # Critic: state value estimate
        value = self.critic(features)

        return action_mean, action_std, value

    def get_action(self, state, action=None):
        action_mean, action_std, value = self.forward(state)

        # Create a Normal distribution with the predicted mean and std
        dist = Normal(action_mean, action_std)

        # If action is provided, use it; otherwise, sample from the distribution
        if action is None:
            action = dist.sample()

        # Compute log probability of the action
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Apply tanh to bound the actions between -1 and 1, then scale to max_action
        action_tanh = torch.tanh(action)

        # We need to account for the change of variables when using tanh
        # This is a correction term for the log probability
        log_prob -= torch.sum(
            torch.log(1 - action_tanh.pow(2) + 1e-6), dim=-1, keepdim=True
        )

        # Scale action to desired range
        scaled_action = action_tanh * self.max_action

        return scaled_action, log_prob, dist.entropy().sum(-1, keepdim=True), value


class PPO:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor_critic = ActorCritic(state_dim, action_dim, max_action)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LR)

        # Rollout buffer
        self.reset_buffers()

    def reset_buffers(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                # During evaluation, just use the mean action
                action_mean, _, _ = self.actor_critic(state)
                return (
                    torch.tanh(action_mean).detach().numpy().flatten()
                    * self.actor_critic.max_action
                )
            else:
                # During training, sample from the distribution
                action, log_prob, _, value = self.actor_critic.get_action(state)

                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(log_prob)
                self.values.append(value)

                return action.detach().numpy().flatten()

    def store_transition(self, reward, done):
        self.rewards.append(torch.FloatTensor([reward]))
        self.dones.append(torch.FloatTensor([done]))

    def train(self):
        # Convert lists to tensors
        states = torch.cat(self.states).detach()
        actions = torch.cat(self.actions).detach()
        old_log_probs = torch.cat(self.log_probs).detach()
        rewards = torch.cat(self.rewards).detach()
        values = torch.cat(self.values).detach()
        dones = torch.cat(self.dones).detach()

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0

        # Get value estimate for the final state
        with torch.no_grad():
            _, _, next_value = self.actor_critic(states[-1])

        # Compute advantages and returns
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO training epochs
        for _ in range(PPO_EPOCHS):
            # Get new action probabilities and values
            _, new_log_probs, entropy, new_values = self.actor_critic.get_action(
                states, actions
            )

            # Compute ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages

            # Actor loss (policy loss)
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (value function loss)
            critic_loss = F.mse_loss(new_values, returns)

            # Entropy bonus (to encourage exploration)
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                actor_loss + CRITIC_DISCOUNT * critic_loss + ENTROPY_BETA * entropy_loss
            )

            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffers
        self.reset_buffers()

        return actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def save(self, filename):
        torch.save(self.actor_critic.state_dict(), filename + "_ppo")
        torch.save(self.optimizer.state_dict(), filename + "_ppo_optimizer")

    def load(self, filename):
        self.actor_critic.load_state_dict(torch.load(filename + "_ppo"))
        self.optimizer.load_state_dict(torch.load(filename + "_ppo_optimizer"))
