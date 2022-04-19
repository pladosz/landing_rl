'''
Recurrent Deterministic Policy Gradient (DDPG with GRU network)
Update with batch of episodes for each time, so requires each episode has the same length.
'''

import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

from common.buffers import *
from common.utils import *
from common.rdpg import *
from common.evaluator import *

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import argparse
from gym import spaces

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#pybullet imports
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

#class NormalizedActions(gym.ActionWrapper): # gym env wrapper
#    def _action(self, action):

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RDPG')
    # Set Environment
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_gazebo2_px4:px4-landing-v1', type=str, help='open-ai gym environment')

    # Set network parameter
    parser.add_argument('--hidden_1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden_2', default=64, type=int, help='input num of GRU layer')
    parser.add_argument('--hidden_3', default=64, type=int, help='output num of GRU layer')
    parser.add_argument('--n_layers', default=1, type=int, help='number of stack for hidden layer')
    parser.add_argument('--rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.00001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.95, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.0001, type=float, help='moving average for target network')
    parser.add_argument('--drop_prob', default=0.2, type=float, help='dropout_probability')

    # Set learning parameter
    parser.add_argument('--rbsize', default=20000, type=int, help='Memory size')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--blength', default=1, type=int, help='minibatch sequence length')
    parser.add_argument('--warmup', default=20000, type=int, help='warmup size (steps or episodes)')
    parser.add_argument('--max_episodes', default=100000, type=int, help='Number of episodes')
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode')
    parser.add_argument('--validate_episodes', default=5, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_interval', default=1000, type=int, help='Validation episode interval')
    parser.add_argument('--epsilon_rate', default=0.1, type=float, help='linear decay of exploration policy')

    #etc
    parser.add_argument('--pause_time', default=0, type=float, help='Pause time for evaluation')
    parser.add_argument('--model_path', default='model/rdpg_pf_dir_vel_no_penalty_center_extreme_noisy_1004', type=str, help='Output root')
    parser.add_argument('--model_path_current', default='model/rdpg_pf_dir_vel_no_penalty_center_extreme_noisy_1004_current', type=str, help='Output root')
    parser.add_argument('--model_path_checkpoint', default='model/rdpg_landing_checkpoint.pt', type=str, help='checkpoint save file')
    parser.add_argument('--save_interval', default=100, type=int, help='how often checkpoint is saved')
    parser.add_argument('--load_checkpoint', default = '', type = str, help= 'string for loading data')


    args = parser.parse_args()


    env = gym.make(args.env)
    action_space = env.action_space
    state_space  = env.observation_space
    # hidden_dim = args.hidden
    # explore_steps = 0  # for random exploration
    batch_size = args.bsize  # each sample in batch is an episode for gru policy (normally it's timestep)
    update_itr = 1  # update iteration
    n_layers = args.n_layers
    replay_buffer_size = args.rbsize
    replay_buffer = ReplayBufferGRU(replay_buffer_size)
    args.model_path = get_output_folder(args.model_path, args.env)
    args.model_path_current = get_output_folder(args.model_path_current, args.env)
    torch.autograd.set_detect_anomaly(True)

    alg = RDPG(args, replay_buffer, state_space, action_space)
    evaluate = Evaluator(args)

    
    if args.mode == 'train':
        # alg.load_model(model_path)
#        alg.load_model('model/rdpg_pf_seq_1/gym_ste:StePFilterConvHardEnv-v0-run9')
        # hyper-parameters
        max_episodes  = args.max_episodes
        max_steps   = args.max_episode_length
        frame_idx   = 0
        total_steps = 0
        epsilon_steps = int(max_episodes*max_steps*args.epsilon_rate)
        rewards=[]
        highest_reward = -100
        update_bool = False
        validate_start = 0
        starting_episode = 0
        if args.load_checkpoint:
            checkpoint = torch.load(args.load_checkpoint)
            epsilon_steps = checkpoint['epsilon']
            alg.load_checkpoint(checkpoint)
            starting_episode = checkpoint['episode']
            total_steps =  checkpoint['total_steps']
            frame_idx =  checkpoint['frame_idx']
            batch_length = checkpoint['batch_length']
        for i_episode in range (starting_episode, max_episodes):
            q_loss_list=[]
            policy_loss_list=[]
            state = env.reset()
            episode_reward = 0
            last_action = [0, 0, 0, 0]
#            episode_state = []
#            episode_action = []
#            episode_last_action = []
            episode_reward = []
#            episode_next_state = []
#            episode_done = []
            hidden_out = torch.zeros([n_layers, 1, args.hidden_3], dtype=torch.float).cuda()
            # initialize hidden state for gru, (hidden, cell), each is (layer, batch, dim)

            batch_length = 0

            if update_bool == False and len(replay_buffer) == args.rbsize:
                update_bool = True
                validate_start = i_episode
            
            if evaluate is not None and (i_episode-validate_start)%args.validate_interval == 0 and update_bool:
                policy = lambda x, h_in: alg.policy_net.get_action(x, h_in, 0.)
                debug = True
                visualize = True
                validate_reward = evaluate(env, policy, i_episode, debug, visualize)
                if debug:
                    prRed('[Evaluate] Episode_{:07d}: mean_reward:{}'.format(i_episode, validate_reward))
                # Save intermediate model
                if highest_reward < validate_reward:
                    prRed('Highest reward: {}, Validate_reward: {}'.format(highest_reward, validate_reward))
                    highest_reward = validate_reward
                    alg.save_model(args.model_path)
            
            for step in range(max_steps):
                hidden_in = hidden_out
                noise_level = max((epsilon_steps - total_steps)/epsilon_steps, 0)
                action, hidden_out = alg.policy_net.get_action(state, hidden_in, noise_level)
                next_state, reward, done, info = env.step(action)
                if batch_length==0:
                    batch_state = []
                    batch_action = []
                    batch_last_action = []
                    batch_reward = []
                    batch_next_state = []
                    batch_done = []
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out

                batch_state.append(state)
                batch_action.append(action)
                batch_last_action.append(last_action)
                batch_reward.append(reward)
                batch_next_state.append(next_state)
                batch_done.append(done)

                episode_reward.append(reward)

#                print(len(ini_hidden_in))

#                replay_buffer.push(hidden_in, hidden_out, [state], [action], [last_action], [reward], [next_state], [done])

                state = next_state
                last_action = action
                frame_idx += 1
                total_steps += 1
                batch_length += 1

                #env.render(mode='human')


                if batch_length == args.blength:
                    batch_length = 0
                    replay_buffer.push(ini_hidden_in, ini_hidden_out, batch_state, batch_action, batch_last_action, \
                                       batch_reward, batch_next_state, batch_done)
#                    replay_buffer.get_length()

                if update_bool:
                    for _ in range(update_itr):
                        q_loss, policy_loss = alg.update(batch_size)
                        q_loss_list.append(q_loss)
                        policy_loss_list.append(policy_loss)
                        alg.save_model(args.model_path_current)

                if done:  # should not break for gru cases to make every episode with same length
                    batch_length = 0
                    break        


            print('Eps: ', i_episode, '| Reward: ', np.sum(episode_reward), '| Loss: ', np.average(q_loss_list), np.average(policy_loss_list), 
                  " | total_step: ", total_steps, " | buffer length: ", replay_buffer.get_length())
            if i_episode % args.save_interval == 0:
                alg.save_checkpoint(args.model_path_checkpoint, i_episode,  np.sum(episode_reward), epsilon_steps, total_steps, frame_idx, batch_length)
#            replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, \
#                episode_reward, episode_next_state, episode_done)

#            rewards.append(np.sum(episode_reward))
    
    if args.mode == 'test':
        test_episodes = 50
        max_steps=300
        alg.load_model('model/rdpg_pf_center_very_hard_current/gym_ste:StePFilterConvCenterVeryHardEnv-v0-run1')
        #alg.load_model(model_path)
        total_reward = 0

        for episode in range(test_episodes):
            # reset at the start of episode
            env.close()
            observation = env.reset()
            #env.render_background(mode='human')
            episode_steps = 0
            episode_reward = 0.
            
            done = False

            hidden_out = torch.zeros([n_layers, 1, args.hidden_3], dtype=torch.float).cuda()
            while not done:
                hidden_in = hidden_out
                # basic operation, action ,reward, blablabla ...
#                action, hidden_out = policy(observation, hidden_in)
                action, hidden_out= alg.policy_net.get_action(observation, hidden_in, noise_scale=0.0)  # no noise for testing
#                print("current action:" + str(action))
                observation, reward, done, info = env.step(action)
#                episode_memory.append(observation)
#                observation = episode_memory.getObservation(self.window_length, observation)
                # Change the episode when episode_steps reach max_episode_length
                
                #env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1
                time.sleep(args.pause_time)
            
            prRed('[Evaluate] Episode_{:07d}: mean_reward:{}'.format(episode, episode_reward))
            total_reward += episode_reward

        mean_reward = total_reward/test_episodes
        prGreen('[Evaluate] Episode_{:07d}: mean_reward:{}'.format(episode, mean_reward))

