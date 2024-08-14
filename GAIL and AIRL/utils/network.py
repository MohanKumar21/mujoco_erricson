import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def build_mlp(input_dim, output_dim, hidden_units = [64, 64], hidden_activation = nn.Tanh(), output_activation = None):
    layers = []
    units = input_dim

    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units

    layers.append(nn.Linear(units, output_dim))

    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5*noises.pow(2) - log_stds).sum(dim = -1, keepdim = True) - 0.5*math.log(2*math.pi) * log_stds.size(-1)
    return gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim = -1, keepdim = True)  

def reparameterize(means, log_stds):
    '''
    The reparametrization trick allows us to rewrite the epectation over actions (where the distribution depends
    on policy parameters) into an expectation over noise (now the distribution has no dependednce on parameters)
    '''
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)

def atanh(x):
    # 1e-6 for numerical stability
    return 0.5 * ( torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6) )

def evaluate_log_pi(means, log_stds, actions):
    '''
    Evaluate the log probability of actions under a Gaussian policy
    '''
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

class StateIndependentPolicy(nn.Module):
    
    def __init__(self, state_shape, action_shape, hidden_units = (64,64), hidden_activation = nn.Tanh()):
        super().__init__()

        self.net = build_mlp(state_shape[0], action_shape[0], hidden_units, hidden_activation)
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))
    
    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)
    
    def evaluate_log_pi(self, states, actions):
        
        return evaluate_log_pi(self.net(states), self.log_stds, actions)
    

class StateDependentPolicy(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units = (256, 256), hidden_activation = nn.ReLU(inplace = True)):
        super().__init__()

        self.net = build_mlp(state_shape[0], 2*action_shape[0], hidden_units, hidden_activation)

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim = -1)[0])
    
    def sample(self, states):
        # print("zendeya")
        # print("states", states.shape)
        # print("net architecture", self.net)
        means, log_stds = self.net(states).chunk(2, dim=-1)
        # print("log_stds", log_stds.shape)s
        log_stds = log_stds.clamp(-20, 2)
        return reparameterize(means, log_stds)


class TwinnedStateActionFunction(nn.Module):
    def __init__(self, state_shape, action_shape, hidden_units = (256,256), hidden_activation = nn.ReLU(inplace=True)):
        super().__init__()

        self.net1 = build_mlp(
            state_shape[0] + action_shape[0], 1, hidden_units, hidden_activation
        )

        self.net2 = build_mlp(
            state_shape[0] + action_shape[0], 1, hidden_units, hidden_activation
        )

    def forward(self, states, actions):
        xs = torch.cat([states, actions], dim = -1)
        return self.net1(xs), self.net2(xs)
    
    def Q1(self, states, actions):
        xs = torch.cat([states, actions], dim = -1)
        return self.net1(xs)
    
class StateFunction(nn.Module):

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return self.net(states)

class AIRLDiscriminator(nn.Module):
    def __init__(self, state_shape, gamma, 
                 hidden_units_r = (64,64),
                 hidden_units_v = (64,64),
                 hidden_activation_r = nn.ReLU(inplace = True),
                 hidden_activation_v = nn.ReLU(inplace = True)):
        super().__init__()

        self.g = build_mlp(state_shape[0], 1, hidden_units_r, hidden_activation_r)
        self.h = build_mlp(state_shape[0], 1, hidden_units_v, hidden_activation_v)
        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + (1 - dones) * self.gamma * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        return self.f(states, dones, next_states) - log_pis
    
    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)
        










