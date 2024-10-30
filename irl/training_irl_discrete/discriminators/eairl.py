# from discriminators.base import Discriminator
# from networks.discriminator_network import Q_phi, Empowerment, Reward

# import torch
# import torch.nn as nn

# class EAIRL(Discriminator):
#     def __init__(self, writer, device, state_dim, action_dim, args):
#         super(EAIRL, self).__init__()
#         self.writer = writer
#         self.device = device
#         self.args = args

#         # Network initialization adapted for discrete states/actions in Taxi-v3
#         self.q_phi = Q_phi(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim,
#                            self.args.activation_function, self.args.last_activation, self.args.trainable_std)

#         # Empowerment and reward functions
#         self.empowerment = Empowerment(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim,
#                                        self.args.activation_function, self.args.last_activation)
#         self.empowerment_t = Empowerment(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim,
#                                          self.args.activation_function, self.args.last_activation)
#         self.reward = Reward(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim,
#                              self.args.activation_function, self.args.last_activation)

#         # Synchronize empowerment target network
#         self.empowerment_t.load_state_dict(self.empowerment.state_dict())

#         self.criterion = nn.BCELoss()
#         self.q_phi_optimizer = torch.optim.Adam(self.q_phi.parameters(), lr=self.args.lr)
#         self.empowerment_optimizer = torch.optim.Adam(self.empowerment.parameters(), lr=self.args.lr)
#         self.reward_optimizer = torch.optim.Adam(self.reward.parameters(), lr=self.args.lr)

#         self.network_init()
#         self.mse = nn.MSELoss()
#         self.iter = 0

#     def encode_state(self, state):
#         # Convert discrete states to one-hot encoding
#         state_one_hot = torch.nn.functional.one_hot(state, num_classes=self.args.state_dim).float()
#         return state_one_hot

#     def encode_action(self, action):
#         # Convert discrete actions to one-hot encoding
#         action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.args.action_dim).float()
#         return action_one_hot

#     def get_d(self, state, next_state, action, done_mask, log_prob):
#         exp_f = torch.exp(self.get_f(state, next_state, action, done_mask))
#         return exp_f / (exp_f + torch.exp(log_prob))

#     def get_f(self, state, next_state, action, done_mask):
#         print(self.reward(state, action))
#         val = self.reward(state, action) + \
#               done_mask.to(self.device) * (self.args.gamma * self.empowerment_t(next_state) - self.empowerment(state))
#         return val

#     def get_reward(self, log_prob, state, action, next_state, done):
#         done_mask = 1 - done.float()
#         return (self.get_f(state, next_state, action, done_mask) - log_prob -
#                 self.args.i_lambda * self.get_loss_i(state, next_state, action, log_prob)).detach()

#     def get_loss_q(self, state, next_state, action):
#         # For discrete actions, treat the output of Q_phi as logits
#         logits = self.q_phi(state, next_state)
#         loss = self.mse(logits, action)
#         return loss

#     def get_loss_i(self, state, next_state, action, log_prob):
#         logits = self.q_phi(state, next_state)
#         q_log_prob = torch.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1).detach()
#         approx_1 = self.args.beta * q_log_prob
#         approx_2 = log_prob + self.empowerment(state)
#         loss = self.mse(approx_1, approx_2)
#         return loss

#     def forward(self, log_prob, state, action, next_state, done_mask):
#         return self.get_d(state, next_state, action, done_mask, log_prob)

#     def train_network(self, writer, n_epi, agent_s, agent_a, agent_next_s, agent_log_prob, agent_done_mask,
#                       expert_s, expert_a, expert_next_s, expert_log_prob, expert_done_mask):
#         # Encode states and actions to work with discrete space
#         agent_s = self.encode_state(agent_s)
#         agent_a = self.encode_action(agent_a)
#         agent_next_s = self.encode_state(agent_next_s)

#         expert_s = self.encode_state(expert_s)
#         expert_a = self.encode_action(expert_a)
#         expert_next_s = self.encode_state(expert_next_s)

#         # Calculate Q_phi loss
#         loss_q = self.get_loss_q(agent_s, agent_next_s, agent_a)
#         self.q_phi_optimizer.zero_grad()
#         loss_q.backward()
#         self.q_phi_optimizer.step()

#         # Calculate I_loss
#         loss_i = self.get_loss_i(agent_s, agent_next_s, agent_a, agent_log_prob)
#         self.empowerment_optimizer.zero_grad()
#         loss_i.backward()
#         self.empowerment_optimizer.step()

#         # Expert and agent loss
#         expert_preds = self.forward(expert_log_prob, expert_s, expert_a, expert_next_s, expert_done_mask)
#         expert_loss = self.criterion(expert_preds, torch.ones(expert_preds.shape[0], 1).to(self.device))

#         agent_preds = self.forward(agent_log_prob, agent_s, agent_a, agent_next_s, agent_done_mask)
#         agent_loss = self.criterion(agent_preds, torch.zeros(agent_preds.shape[0], 1).to(self.device))

#         # Total reward loss
#         reward_loss = expert_loss + agent_loss
#         self.reward_optimizer.zero_grad()
#         reward_loss.backward()
#         self.reward_optimizer.step()

#         # Logging to TensorBoard
#         if writer is not None:
#             writer.add_scalar("loss/discriminator_loss_q", loss_q.item(), n_epi)
#             writer.add_scalar("loss/discriminator_loss_i", loss_i.item(), n_epi)
#             writer.add_scalar("loss/discriminator_reward_loss", reward_loss.item(), n_epi)

#         if self.iter % self.args.update_cycle == 0:
#             self.empowerment_t.load_state_dict(self.empowerment.state_dict())
#         self.iter += 1


from discriminators.base import Discriminator
from networks.discriminator_network import Q_phi, Empowerment, Reward

import torch
import torch.nn as nn

    
class EAIRL(Discriminator):
    def __init__(self,writer, device, state_dim, action_dim, args):
        super(EAIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.q_phi = Q_phi(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function,self.args.last_activation,self.args.trainable_std)
        self.empowerment = Empowerment(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function ,self.args.last_activation)
        self.empowerment_t = Empowerment(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function ,self.args.last_activation)
        self.reward = Reward(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function , self.args.last_activation)
        self.empowerment_t.load_state_dict(self.empowerment.state_dict())
        self.criterion = nn.BCELoss()
        self.q_phi_optimizer = torch.optim.Adam(self.q_phi.parameters(), lr=self.args.lr)
        self.empowerment_optimizer = torch.optim.Adam(self.empowerment.parameters(), lr=self.args.lr)
        self.reward_optimizer = torch.optim.Adam(self.reward.parameters(), lr=self.args.lr)
        self.network_init()
        self.mse = nn.MSELoss()
        
        self.iter = 0
    def get_d(self,state,next_state,action,done_mask,log_prob):
        exp_f = torch.exp(self.get_f(state,next_state,action,done_mask))
        return exp_f / (exp_f+torch.exp(log_prob)) 
    
    def get_f(self,state,next_state,action,done_mask):
        val= self.reward(state,action) +\
    done_mask.to(self.device) * (self.args.gamma * self.empowerment_t(next_state) - self.empowerment(state))
        return val 

    def get_reward(self,log_prob,state,action,next_state,done):
        done_mask = 1 - done.float()
        return (self.get_f(state,next_state,action,done_mask) - log_prob - self.args.i_lambda * self.get_loss_i(state,next_state,action,log_prob)).detach() 
    def get_loss_q(self,state,next_state,action):
        mu,sigma = self.q_phi(state,next_state)
        loss = self.mse(mu,action)
        return loss
    
    def get_loss_i(self,state,next_state,action,log_prob):
        mu,sigma = self.q_phi(state,next_state)
        dist = torch.distributions.Normal(mu,sigma)
        q_log_prob = dist.log_prob(action).sum(-1,keepdim=True).detach()
        approx_1 = self.args.beta * q_log_prob
        approx_2 = log_prob + self.empowerment(state)
        loss = self.mse(approx_1,approx_2)
        return loss
    
    def forward(self,log_prob,state,action,next_state,done_mask):
        return self.get_d(state,next_state,action,done_mask,log_prob)
        
    def train_network(self,writer,n_epi,agent_s,agent_a,agent_next_s,\
                      agent_log_prob,agent_done_mask,expert_s,expert_a,expert_next_s,\
                      expert_log_prob,expert_done_mask):
        
        loss_q = self.get_loss_q(agent_s,agent_next_s,agent_a)
        self.q_phi_optimizer.zero_grad()
        loss_q.backward()
        self.q_phi_optimizer.step()
        
        loss_i = self.get_loss_i(agent_s,agent_next_s,agent_a,agent_log_prob)
        self.empowerment_optimizer.zero_grad()
        loss_i.backward()
        self.empowerment_optimizer.step()
        
        expert_preds = self.forward(expert_log_prob,expert_s,expert_a,expert_next_s,expert_done_mask)
        expert_loss = self.criterion(expert_preds,torch.ones(expert_preds.shape[0],1).to(self.device)) 
        
        agent_preds = self.forward(agent_log_prob,agent_s,agent_a,agent_next_s,agent_done_mask)
        agent_loss = self.criterion(agent_preds,torch.zeros(agent_preds.shape[0],1).to(self.device))
        
        reward_loss = expert_loss+agent_loss
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()
        
        if writer != None:
            writer.add_scalar("loss/discriminator_loss_q", loss_q.item(), n_epi)
            writer.add_scalar("loss/discriminator_loss_i", loss_i.item(), n_epi)
            writer.add_scalar("loss/discriminator_reward_loss", reward_loss.item(), n_epi)
        if self.iter % self.args.update_cycle == 0:
            self.empowerment_t.load_state_dict(self.empowerment.state_dict())
        self.iter += 1