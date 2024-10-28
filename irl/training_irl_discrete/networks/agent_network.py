from networks.base import Network

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None, trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        def forward(self, x):
            logits = self._forward(x)
            return logits

class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)

    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)