import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
import numpy as np
import copy

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=None):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim)
        )

        self.action_bound = action_bound  # Max action value the environment can accept

    def forward(self, x):
        return torch.tanh(self.net(x)) * self.action_bound

class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=None):
        super(QValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)   # Concatenate state and action
        return self.net(cat)

class TD3:
    ''' TD3 algorithm '''
    def __init__(self, state_dim, action_dim, action_bound, policy_noise, noise_clip, actor_lr, critic_lr, tau, gamma, device, delay, lr_step_size, lr_gamma, hidden_dim=None):
        self.actor = PolicyNet(state_dim, action_dim, action_bound).to(device)
        self.critic_1 = QValueNet(state_dim, action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, action_dim).to(device)

        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)

        # Disable gradient computation for all target network parameters
        for param in self.target_actor.parameters():
            param.requires_grad = False
        for param in self.target_critic_1.parameters():
            param.requires_grad = False
        for param in self.target_critic_2.parameters():
            param.requires_grad = False

        # Define optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.scheduler = StepLR(self.actor_optimizer, step_size=lr_step_size, gamma=lr_gamma)

        self.action_bound = action_bound
        self.gamma = gamma
        self.policy_noise = policy_noise  # Gaussian noise std dev, mean set to 0
        self.noise_clip = noise_clip  # Noise clip upper bound
        self.tau = tau  # Target network soft update parameter
        self.action_dim = action_dim
        self.device = device
        self.delay = delay

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = np.array(self.actor(state).tolist())                       # Convert output tensor to 3-D ndarray
        return action   # Output is 3-D ndarray
    

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)   # Soft update: tau=1 means full copy to target network

    def update(self, transition_dict, env_i):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device) 
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (self.target_actor(next_states) + noise).clamp(-self.action_bound, self.action_bound)

        # next_actions = self.target_actor(next_states) # Without noise
        next_q_values_1 = self.target_critic_1(next_states, next_actions)
        next_q_values_2 = self.target_critic_2(next_states, next_actions)
        next_q_values = torch.min(next_q_values_1,next_q_values_2)

        q_targets = rewards + (1-dones) * self.gamma * next_q_values
        # q_targets = rewards + self.gamma * next_q_values

        critic_1_loss = F.mse_loss(self.critic_1(states, actions), q_targets) # Critic output vs target Q-value computed from target networks
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        critic_2_loss = F.mse_loss(self.critic_2(states, actions), q_targets) # Critic output vs target Q-value computed from target networks
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if env_i % self.delay == 0:
            actor_loss = -torch.mean(self.critic_1(states, self.actor(states))) # Maximize action score from critic
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)  # Soft update policy network
            self.soft_update(self.critic_1, self.target_critic_1)  # Soft update value network
            self.soft_update(self.critic_2, self.target_critic_2)  # Soft update value network

    def save_model(self, path):
        # Save weights for four networks
        torch.save(self.actor.state_dict(), path + '/actor.pth')
        torch.save(self.critic_1.state_dict(), path + '/critic_1.pth')
        torch.save(self.critic_2.state_dict(), path + '/critic_2.pth')
        torch.save(self.target_actor.state_dict(), path + '/target_actor.pth')
        torch.save(self.target_critic_1.state_dict(), path + '/target_critic_1.pth')
        torch.save(self.target_critic_2.state_dict(), path + '/target_critic_2.pth')

    def load_model(self, path):
        # Load weights for four networks
        self.actor.load_state_dict(torch.load(path + '/actor.pth'))
        self.critic_1.load_state_dict(torch.load(path + '/critic_1.pth'))
        self.critic_2.load_state_dict(torch.load(path + '/critic_2.pth'))
        self.target_actor.load_state_dict(torch.load(path + '/target_actor.pth'))
        self.target_critic_1.load_state_dict(torch.load(path + '/target_critic_1.pth'))
        self.target_critic_2.load_state_dict(torch.load(path + '/target_critic_2.pth'))
    
    def load_model_cpu(self, path):
        # Load weights for four networks
        self.actor.load_state_dict(torch.load(path + '/actor.pth',map_location=torch.device('cpu')))
        self.critic_1.load_state_dict(torch.load(path + '/critic_1.pth',map_location=torch.device('cpu')))
        self.critic_2.load_state_dict(torch.load(path + '/critic_2.pth',map_location=torch.device('cpu')))
        self.target_actor.load_state_dict(torch.load(path + '/target_actor.pth',map_location=torch.device('cpu')))
        self.target_critic_1.load_state_dict(torch.load(path + '/target_critic_1.pth',map_location=torch.device('cpu')))
        self.target_critic_2.load_state_dict(torch.load(path + '/target_critic_2.pth',map_location=torch.device('cpu')))