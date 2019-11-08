"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""
# import os, sys, random
import tensorflow as tf
import numpy as np

import time


#####################  hyper parameters  ####################

MAX_EPISODES = 500
MAX_EP_STEPS = 200
LR_A = 0.00005   # learning rate for actor
LR_C = 0.0005    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64

RENDER = False

# script_name = os.path.basename(__file__)

###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim  + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY-1, size=BATCH_SIZE)
        # bt = self.memory[indices, :]
        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]
        bs,ba,br,bs_ = [], [], [], []
        for i in indices:
            s=self.memory[i,:self.s_dim]
            a=self.memory[i,self.s_dim:self.s_dim+self.a_dim]
            r=self.memory[i,-1]
            s1=self.memory[i+1,:self.s_dim]
            bs.append(np.array(s,copy=False))
            ba.append(np.array(a,copy=False))
            br.append(np.array(r,copy=False))
            bs_.append(np.array(s1,copy=False))
        bs=np.reshape(bs,(BATCH_SIZE,self.s_dim))
        ba=np.reshape(ba,(BATCH_SIZE,self.a_dim))
        br=np.reshape(br,(BATCH_SIZE,1))
        bs_=np.reshape(bs_,(BATCH_SIZE,self.s_dim))
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r):
        transition = np.hstack((s, a, [r]))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 128, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 128
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

class SchedulerTrainer(object):
    def __init__(self, s_dim, a_dim):
        self.trainer = DDPG(a_dim, s_dim, 1)

def bound(input,x_min,x_max,y_min,y_max):
    if np.isnan(input):
        print("input is nan")
    result=y_min+(y_max-y_min)/(x_max-x_min)*(input-x_min)
    return (result)


###############################  training  ####################################

# env = Env(10000)
# env.seed(1)

# s_dim = env.s_dim
# a_dim = env.a1_dim+env.a_dim
# a_bound = 1

# ddpg = DDPG(a_dim, s_dim, a_bound)

# var = 3  # control exploration
# t1 = time.time()
# for i in range(MAX_EPISODES):
#     env.reset()
#     ep_reward = 0
#     reward=None
#     done=False
#     nolegal_action=0
#     while True:
#         env.run()
#         for gpu in env.BaseStation.gpu_cluster:
#             if gpu.busy_flag is False and gpu.batch_flag is False:
#                 state=env.get_state()
#                 action=ddpg.choose_action(state)
#                 action = np.clip(np.random.normal(action, var), -1, 1)    # add randomness to action selection for exploration
#                 new_action=translate_action(action)
#                 # print(state)
#                 # print("a"+str(env.sys_time))
#                 # print(new_action)
#                 if env.Is_action_legal(new_action) is False:
#                     nolegal_action+=1
#                     reward=-10000
#                     print("r:"+str(reward))
#                     ep_reward+=reward
#                     ddpg.store_transition(state, action, reward )
#                 else:
#                     env.wait_todo.append([gpu.id,new_action])
#                     env.proceed(True,gpu.id,new_action) 
#         for gpu in env.BaseStation.gpu_cluster:
#             reward,flag=env.step(gpu)
#             if flag:
#                 new_action=gpu.action
#                 env.wait_todo.remove([gpu.id,new_action])
#                 if gpu.batch.real_size==pow(2,gpu.batch.size_log):
#                     ep_reward+=reward
#                     ddpg.store_transition(state, action, reward )
#         env.stop_flag=env.Is_done()  
#         env.sys_time += 1  

#         if ddpg.pointer > MEMORY_CAPACITY:
#             var *= .9995    # decay the action randomness
#             ddpg.learn()

        
#         if env.stop_flag:
#             ave_r=env.get_whole_avr_qos()
#             user_ave_qos=env.get_user_avr_qos()
#             fw = open('aver_reward2.txt', 'a')
#             fw.write( str(ave_r)+' '+str(user_ave_qos[0])+' '+str(user_ave_qos[1])+' '+str(user_ave_qos[2])+'\n')
#             fw.close()
#             print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, 'ave_r:', ave_r, '|', user_ave_qos[0], '|', user_ave_qos[1], '|', user_ave_qos[2], )
#             # if ep_reward > -300:RENDER = True
#             break

# print('Running time: ', time.time() - t1)
