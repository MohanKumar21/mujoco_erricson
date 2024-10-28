from agents.ppo    import PPO
from agents.agent            import Agent


from discriminators.eairl    import EAIRL
from utils.utils             import RunningMeanStd, Dict, make_transition

from configparser            import ConfigParser
from argparse                import ArgumentParser

import os
import gymnasium as gym
import numpy as np

import torch
from torch.autograd import Variable

os.makedirs('./model_weights', exist_ok=True)

env = gym.make("Taxi-v3")

action_dim = 1
state_dim = env.observation_space.n
print(action_dim, state_dim, action_dim)
print()
parser = ArgumentParser('parameters')
# Taxi specific 
def decode_positions(states):
    """
    Gets an state from env.render() (int) and returns
    the taxi position (row, col), the passenger position
    and the destination location

    :param states: a list of states represented as integers [0-499]
    :return: taxi_row, taxi_col, pass_code, dest_idx
    """
    dest_loc = [state % 4 for state in states]
    states = [state // 4 for state in states]
    pass_code = [state % 5 for state in states]
    states = [state // 5 for state in states]
    taxi_col = [state % 5 for state in states]
    states = [state // 5 for state in states]
    taxi_row = states
    return taxi_row, taxi_col, pass_code, dest_loc
def encode_states(states, encode_method, state_dim):
    """
        Gets a list of integers and returns their encoding
        as 1 of 2 possible encoding methods:
            - one-hot encoding (array)
            - position encoding

    :param states: list of integers in [0,num_states-1]
    :param encode_method: one of 'one_hot', 'positions'
    :param state_dim: dimension of state (used for 'one_hot' encoding)
    :return: states_encoded: one hot encoding of states
    """

    batch_size = len(states)

    if encode_method is 'positions':
        '''
            position encoding encodes the important game positions as 
            a 19-dimensional vector:
                - 5 dimensions are used for one-hot encoding of the taxi's row (0-4)
                - 5 dimensions are used for one-hot encoding of the taxi's col (0-4)
                - 5 dimensions are used for one-hot encoding of the passenger's position:
                    0 is 'R', 1 is 'G', 2 is 'Y', 3 is 'B' and 4 is if the passenger in the taxi
                - 4 dimensions are used for one-hot encoding of the destination location:
                    0 is 'R', 1 is 'G', 2 is 'Y' and 3 is 'B'
                we simply concatenate those vectors into a 19-dim. vector with 4 ones in it
                corresponding to the positions encoding and the rest are zeros.      
        '''

        taxi_row, taxi_col, pass_code, dest_loc = decode_positions(states)

        # one-hot encode taxi's row
        taxi_row_onehot = np.zeros((batch_size, 5))
        taxi_row_onehot[np.arange(batch_size), taxi_row] = 1
        # one-hot encode taxi's col
        taxi_col_onehot = np.zeros((batch_size, 5))
        taxi_col_onehot[np.arange(batch_size), taxi_col] = 1
        # one-hot encode row
        pass_code_onehot = np.zeros((batch_size, 5))
        pass_code_onehot[np.arange(batch_size), pass_code] = 1
        # one-hot encode row
        dest_loc_onehot = np.zeros((batch_size, 4))
        dest_loc_onehot[np.arange(batch_size), dest_loc] = 1

        states_encoded = np.concatenate([taxi_row_onehot, taxi_col_onehot,
                                         pass_code_onehot, dest_loc_onehot], axis=1)

    else:   # one-hot
        states_encoded = np.zeros((batch_size, state_dim))
        states_encoded[np.arange(batch_size), states] = 1

    return states_encoded
# Define all the model parameters
hidden_units = 32               # num. units in hidden layer
replay_buffer_size = 200000     # buffer size
start_learning = 50000          # num. transitions before start learning
target_update_freq = 10000      # num. transitions between Q_target network updates
eps = 0.1                       # final epsilon for epsilon-greedy action selection
schedule_timesteps = 350000     # num. transitions for epsilon annealing
batch_size = 32                 # size of batch size for training
gamma = 0.99                    # discount factor of MDP
eps_optim = 0.01                # epsilon parameter for optimization (improves stability of optimizer)
alpha = 0.95                    # alpha parameter of RMSprop optimizer
learning_rate = 0.00025         # step size for optimization process


encode_method = 'one_hot'     # state encoding method ('one_hot' or 'positions')

if encode_method is 'positions':
    state_dim = 19  # explained in utils.encode_states
else:   # one-hot
    state_dim = env.observation_space.n

regularization = None           # regularization may be 'regularization'
save_fig = True     # whether to save figure of accumulated reward
save_model = True      # whether to save the DQN model

# Define 2-layered architecture
architecture = {"state_dim": state_dim,
                "hidden_units": hidden_units,
                "num_actions": env.action_space.n}
# Pack the epsilon greedy exploration parameters
exploration_params = {"timesteps": schedule_timesteps, "final_eps": eps}

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 


parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs, (default: 1001)')
parser.add_argument("--agent", type=str, default = 'ppo', help = 'actor training algorithm(default: ppo)')
parser.add_argument("--discriminator", type=str, default = 'eairl', help = 'discriminator training algorithm(default: gail)')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval')
parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: True)')

args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

demonstrations_location_args = Dict(parser,'demonstrations_location',True)
agent_args = Dict(parser,args.agent)
discriminator_args = Dict(parser,args.discriminator)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None


if args.discriminator == 'eairl':
    discriminator = EAIRL(writer, device, state_dim, action_dim, discriminator_args)
else:
    raise NotImplementedError
    
if args.agent == 'ppo':
    algorithm = PPO(device, state_dim, action_dim, agent_args)
else:
    raise NotImplementedError
agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()
    
state_rms = RunningMeanStd(state_dim)
print(state_rms)
score_lst = []
discriminator_score_lst = []
score = 0.0
done=False
terminated=False
discriminator_score = 0
if agent_args.on_policy == True:
    state_lst = []
    state_,_ = (env.reset())
    # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    state=state_
    for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            # print(t,done,terminated)
            if args.render:    
                env.render()
            
            state_lst.append(state_)
            
            state=encode_states([state],encode_method,state_dim)
            state = Variable(torch.from_numpy(state).type(dtype))
            # action, log_prob = agent.get_action(torch.from_numpy(state).float().unsqueeze(0).to(device))
            # action, log_prob = agent.get_action(torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device))
            action,log_prob =agent.get_action(state)
            
            # action=((action.data.argmax())).unsqueeze(0)
            # print(action,log_prob,state)
            next_state_, r, done,terminated, info = env.step(int(action))
            # next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            next_state = next_state_ 
            next_state=encode_states([next_state],encode_method,state_dim)
            next_state = Variable(torch.from_numpy(next_state).type(dtype))
            print(torch.tensor(action).unsqueeze(0).unsqueeze(0))
            if discriminator_args.is_airl:
                # print(action,(next_state).float().to(device))
                reward = discriminator.get_reward(\
                                        log_prob,\
                                        torch.tensor(state).unsqueeze(0).float().to(device),torch.tensor(action).unsqueeze(0).unsqueeze(0).unsqueeze(0),\
                                        torch.tensor(next_state).unsqueeze(0).float().to(device),\
                                                              torch.tensor(done).view(1,1)\
                                                 ).item()
            else:
                reward = discriminator.get_reward(torch.tensor(state).unsqueeze(0).float().to(device),action).item()

            transition = make_transition(state,\
                                         torch.tensor(action),\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += r

            discriminator_score += reward
            if done or terminated:
                state_,_ = (env.reset())
                state=state_
                # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if writer != None:
                    writer.add_scalar("score/real", score, n_epi)
                    writer.add_scalar("score/discriminator", discriminator_score, n_epi)
                score = 0
                discriminator_score = 0
            else:
                state = next_state_
                state_ = next_state_ 
        agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi)
        state_rms.update(np.vstack(state_lst))
        state_lst = []
        
        if n_epi%args.print_interval==0 and n_epi!=0:
            if len(score_lst)==0:
                avg_score=0
            else:
                avg_score=sum(score_lst)/len(score_lst)
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, avg_score))
            score_lst = []
       
        if (n_epi % args.save_interval == 0 )& (n_epi != 0):
                torch.save(agent.state_dict(), './model_weights/model_'+str(n_epi))
        if (n_epi % args.save_interval == 0 )& (n_epi != 0):
                torch.save(discriminator.state_dict(), './model_weights/discriminator_'+str(n_epi))

else : #off-policy
    for n_epi in range(args.epochs):
        score = 0.0
        discriminator_score = 0.0
        state = env.reset()
        done = False
        while not done:
            if args.render:    
                env.render()
            action_, log_prob = agent.get_action(torch.from_numpy(state).float().to(device))
            action = action_.cpu().detach().numpy()
            next_state, r, done, info = env.step(action)
            if discriminator_args.is_airl:
                reward = discriminator.get_reward(\
                            log_prob,
                            torch.tensor(state).unsqueeze(0).float().to(device),action_,\
                            torch.tensor(next_state).unsqueeze(0).float().to(device),\
                                                  torch.tensor(done).unsqueeze(0)\
                                                 ).item()
            else:
                reward = discriminator.get_reward(torch.tensor(state).unsqueeze(0).float().to(device),action_).item()

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 

            state = next_state

            score += r
            discriminator_score += reward
            
            if agent.data.data_idx > agent_args.learn_start_size: 
                agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi, agent_args.batch_size)
        score_lst.append(score)
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
            writer.add_scalar("score/discriminator", discriminator_score, n_epi)
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))