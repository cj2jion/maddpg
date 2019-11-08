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
class SchedulerEnv(object):
    def __init__(self, arglist):
        self.action_list=[]
        self.state_list=[]
       
        self.s_dim=(N_post+N_pre)*arglist.user_num+arglist.user_num
        self.a_dim=2

    def reset(self):
        self.action_list=[]
        self.state_list=[]
       
   
