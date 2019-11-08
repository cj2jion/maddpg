"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""
import argparse
import numpy as np
import tensorflow as tf
import gym
import time
np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error



class Actor(object):
    def __init__(self, sess, arglist, n_features, n_actions):
        self.sess = sess
        with tf.variable_scope("trainer", reuse=tf.AUTO_REUSE):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            self.a = tf.placeholder(tf.int32, None, "act")
            self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

            with tf.variable_scope('Actor',reuse=tf.AUTO_REUSE):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=arglist.num_units,    # number of hidden units
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                )

                self.acts_prob = tf.layers.dense(
                    inputs=l1,
                    units=n_actions,    # output units
                    activation=tf.nn.softmax,   # get action probabilities
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                )

            with tf.variable_scope('exp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a])
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(arglist.lr_a).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, arglist, n_features):
        self.sess = sess
        with tf.variable_scope("trainer", reuse=tf.AUTO_REUSE):
            self.s = tf.placeholder(tf.float32, [1, n_features], "state")
            # self.s=[tf.placeholder(dtype=tf.float32, shape=[n_features, ], name="state"+str(i)) for i in range(len(act_space_n))]
            self.v_ = tf.placeholder(tf.float32, [1,1], "v_next")
            self.r = tf.placeholder(tf.float32, None, 'r')

            # q_input = tf.concat(self.s, 1)
            with tf.variable_scope('Critic',reuse=tf.AUTO_REUSE):
                l1 = tf.layers.dense(
                    inputs=self.s,
                    units=arglist.num_units,  # number of hidden units
                    activation=tf.nn.relu,  # None
                    # have to be linear to make sure the convergence of actor.
                    # But linear approximator seems hardly learns the correct Q.
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                )

                self.v = tf.layers.dense(
                    inputs=l1,
                    units=1,  # output units
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                    bias_initializer=tf.constant_initializer(0.1),  # biases
                )

            with tf.variable_scope('squared_TD_error'):
                self.td_error = self.r + arglist.gamma * self.v_ - self.v
                self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(arglist.lr_c).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

class AgentTrainer(object):
    def __init__(self, name,sess, n_features, n_actions,n_single_features, arglist):
        self.name=name
        self.actor=Actor(sess, arglist, n_features=n_single_features, n_actions=n_actions)
        self.critic=Critic(sess, arglist, n_features=n_features) 


