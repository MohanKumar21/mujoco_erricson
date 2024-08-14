import torch 
import numpy as np 

discriminator_path="model_weights/discriminator_1000"
agent_path="model_weights/model_400"

from agents.ppo    import PPO
from agents.agent            import Agent


from discriminators.eairl    import EAIRL
from utils.utils             import RunningMeanStd, Dict, make_transition

from configparser            import ConfigParser
from argparse                import ArgumentParser

import os
import gym
import numpy as np

import torch

os.makedirs('./model_weights', exist_ok=True)

env = gym.make("Hopper-v4")

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

parser = ArgumentParser('parameters')


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



discriminator.load_state_dict(torch.load(discriminator_path))
if args.agent == 'ppo':
    algorithm = PPO(device, state_dim, action_dim, agent_args)
else:
    raise NotImplementedError
agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()
agent.load_state_dict(torch.load(agent_path))
print(env.observation_space)
state_rms = RunningMeanStd(state_dim)
state,_=env.reset()
rewards_true=[]
reward_predicted=[]
for i in range(5000):
    
    # print(state,action_)
    state = np.clip((state - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    
    action=(np.random.uniform(-1, 1, 3))
    
    # action_, log_prob = agent.get_action(torch.from_numpy(state).float().unsqueeze(0).to(device))
    # action=action_.cpu().numpy()[0]
    action_=torch.tensor(action).unsqueeze(0).float()
    state=torch.tensor(state).unsqueeze(0).float()
    reward_predicted.append(discriminator.reward(state,action_).item())
    print(discriminator.reward(state,action_))
    print("---------------------------------")
    state,reward,done,terminated,info=env.step(action)
    rewards_true.append(reward)
    
    print(reward)
    print("---------------------------------")

def standardize(arr):
    return (arr - arr.mean()) / arr.std()
true_rewards=np.array(rewards_true)
predicted_rewards=np.array(reward_predicted)

def min_max_normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

true_normalized = min_max_normalize(true_rewards)
predicted_normalized = min_max_normalize(predicted_rewards)

true_standardized = standardize(true_rewards)
predicted_standardized = standardize(predicted_rewards)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(true_normalized, predicted_normalized)
print(f"Mean Squared Error: {mse}")


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(true_normalized, predicted_normalized)
print(f"Mean Absolute Error: {mae}")

from scipy.stats import pearsonr

correlation_coefficient, _ = pearsonr(true_standardized, predicted_standardized)
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(true_normalized.reshape(1, -1), predicted_normalized.reshape(1, -1))

print(f"Cosine Similarity: {cosine_sim[0][0]}")
import matplotlib.pyplot as plt

plt.scatter(true_normalized, predicted_normalized)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.savefig("Rewards.jpg")


plt.figure()
plt.hist(true_normalized, bins=20, alpha=0.5, label='True Values')
plt.hist(predicted_normalized, bins=20, alpha=0.5, label='Predicted Values')
plt.legend(loc='upper right')
plt.savefig("rewards_histogram.jpg")

