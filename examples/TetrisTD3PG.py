import Tetris
import gym
from gym import wrappers
import os
import time
import random
import numpy as np
import torch
from TD3PG import TD3PG

class TetrisTD3PG:
    def __init__(self, save_models=True):
        self.env_name = "tetris-v0" # Name of a environment (set it to any Continous environment you want)
        self.seed = 0 # Random seed number
        
        
        self.file_name = "%s_%s_%s" % ("TD3PG", self.env_name, str(self.seed))
        print ("---------------------------------------")
        print ("Settings: %s" % (self.file_name))
        print ("---------------------------------------")
        
        if not os.path.exists("./results"):
            os.makedirs("./results")
        if save_models and not os.path.exists("./pytorch_models"):
            os.makedirs("./pytorch_models")
            
        self.env = gym.make(self.env_name)
        
        
        self.env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        
        
        self.policy = TD3PG(self.state_dim, self.action_dim, self.max_action)
        
        self.evaluations = [self.evaluate_policy()]
        
    def train(self, start_timesteps=1e4, max_timesteps=5e5, eval_freq=5e3,
              save_models=True, expl_noise=0.1, batch_size=100,
              discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):
         # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
         # How often the evaluation step is performed (after how many timesteps)
         # Boolean checker whether or not to save the pre-trained model
         # Exploration noise - STD value of exploration Gaussian noise
         # Size of the batch
         # Discount factor gamma, used in the calculation of the total discounted reward
         # Target network update rate
         # STD of Gaussian noise added to the actions for the exploration purposes
         # Maximum value of the Gaussian noise added to the actions (policy)
         # Number of iterations to wait before the policy network (Actor model) is updated
        
        
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True
        t0 = time.time()
        
        # We start the main loop over 500,000 timesteps
        while total_timesteps < max_timesteps:
            # If the episode is done
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    self.policy.train(self.policy.replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                
                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    self.evaluations.append(self.evaluate_policy())
                    self.policy.save(self.file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (self.file_name), self.evaluations)
    
                # When the training step is done, we reset the state of the environment
                obs = self.env.reset()
                # Set the Done to False
                done = False
                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = self.env.action_space.sample()
            else: # After 10000 timesteps, we switch to the model
                action = self.policy.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=self.env.action_space.shape[0])).clip(self.env.action_space.low, self.env.action_space.high)
            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, _ = self.env.step(action)
            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == self.env.observation_space._max_episode_steps else float(done)
            # We increase the total reward
            episode_reward += reward
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            self.policy.replay_buffer.add((obs, new_obs, action, reward, done_bool))
            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
        # We add the last policy evaluation to our list of evaluations and we save our model
        self.evaluations.append(self.evaluate_policy())
        if save_models: 
            self.policy.save("%s" % (self.file_name), directory="./pytorch_models")
            np.save("./results/%s" % (self.file_name), evaluations)
        
        
    def evaluate_policy(self, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = self.policy.select_action(np.array(obs))
                obs, reward, done, _ = self.env.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward

if "__name__" == "__main__":
    pass