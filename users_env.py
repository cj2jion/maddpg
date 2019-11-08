#-*- coding: UTF-8 -*-

from User import User, update_time, ObT
from BaseStation import BaseStation, Batch
import numpy as np
from math import log, exp
import gym
from gym import spaces
# from multiagent.multi_discrete import MultiDiscrete
N_pre=5
N_post=5
class UserEnv(object):
    def __init__(self, arglist):
        self.a_dim=15
        self.s_dim=44 # 30+1+5*2+3
        self.s_single_dim=22 #10+1+5*2+1
        self.n=arglist.user_num
        self.discrete_action_space=True
        self.action_list=[]
        for i in range(self.n):
            self.action_list.append([])
        self.rew_list=[]
        self.obs_list=[]
        self.single_obs_list=[]
        for i in range(self.n):
            self.obs_list.append([])
            self.single_obs_list.append([])
        self.last_duration_qos=[]
        for i in range(self.n):
            self.last_duration_qos.append([])
        # configure spaces
        self.action_space = spaces.Discrete(self.a_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.s_dim,), dtype=np.float32)
        # for agent in self.User_num:
        #     total_action_space = []
        #     if self.discrete_action_space:
        #         c_action_space = spaces.Discrete(self.a_dim)
        #     else:
        #         c_action_space = spaces.Box(low=0.0, high=1.0, shape=(self.a_dim,), dtype=np.float32)
            
        #     total_action_space.append(c_action_space)
        #     if len(total_action_space) > 1:
        #         # all action spaces are discrete, so simplify to MultiDiscrete action space
        #         if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
        #             act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
        #         else:
        #             act_space = spaces.Tuple(total_action_space)
        #         self.action_space.append(act_space)
        #     else:
        #         self.action_space.append(total_action_space[0])
            # observation space
            # obs_dim = self.s_dim
            # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))


    def reset(self):
        self.action_list=[]
        self.obs_list=[]
        self.single_obs_list=[]
        for i in range(self.n):
            self.obs_list.append([])
            self.single_obs_list.append([])
        self.last_duration_qos=[]
        for i in range(self.n):
            self.last_duration_qos.append([])
   
    def step(self,action_n):
        obs_n = []
        reward_n = []
        done_n = []
         # set action for each agent
        for i in range(self.User_num):
            self._set_action(action_n[i], self.action_space[i])
        
        # record observation for each agent
        for i in range(self.User_num):
            obs_n.append(self._get_obs(i,action_n[i]))
            reward_n.append(self._get_reward(i,action_n[i]))
            done_n.append(self._get_done(i,action_n[i]))

         # all agents get total reward in cooperative case
        reward = np.sum(reward_n)

        return obs_n, reward_n, done_n

   
