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
import pickle
import os

from User import User, update_time, ObT
from users_env import UserEnv
from scheduler_env import SchedulerEnv
from maac import AgentTrainer
from ddpg import SchedulerTrainer
from simulation_env import SimulationEnv
import time
np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    # parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    # parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--num_adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--user_num", type=int, default=3, help="number of user")
    # parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--lr_c", type=float, default=0.0005, help="learning rate for Adam optimizer")
    parser.add_argument("--lr_a", type=float, default=0.00005, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=10000, help="number of episodes to optimize at the same time")
    parser.add_argument("--num_units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp_name", type=str, default="maac", help="name of the experiment")
    parser.add_argument("--save_dir", type=str, default="model/", help="directory in which training state and model should be saved")
    parser.add_argument("--save_rate", type=int, default=50, help="save model once every time this many episodes are completed")
    parser.add_argument("--load_dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--mode", type=str, default="train", help="train or test")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark_iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark_dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots_dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()

def save_state(sess, fname, saver=None):
    """Save all the variables in the current session to the location <fname>"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if saver is None:
        saver = tf.train.Saver()
    saver.save(sess, fname)
    return saver

def get_trainers(env, sess,arglist):
    trainers = []
    trainer=AgentTrainer
    for i in range(env.n):
        trainers.append(trainer(
            "agent_%d" % i, sess, env.s_dim, env.a_dim, env.s_single_dim, arglist))
   
    return trainers

def train(arglist):
    sess = tf.Session()
    userenv=UserEnv(arglist)
    schedulerenv=SchedulerEnv(arglist)
    simulationenv=SimulationEnv(userenv, schedulerenv, arglist)

    # get user agent 
    # obs_shape_n = [userenv.observation_space[i].shape for i in range(userenv.n)]
    trainers = get_trainers(userenv, sess, arglist)
    saver = tf.train.Saver()
    #get scheduler agent
    schedulerTrainer=SchedulerTrainer(schedulerenv.s_dim,schedulerenv.a_dim)
    
    if arglist.mode == "test":
        
        checkpoint = tf.train.get_checkpoint_state("model/")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        
        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
    
        initialize_op = tf.variables_initializer(uninitialized_vars)
        sess.run(initialize_op)
    else:
        sess.run(tf.global_variables_initializer())


    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(userenv.n)]  # individual agent reward
    final_ep_rewards = []  # sum of rewards for training curve
    final_ep_ag_rewards = []  # agent rewards for training curve
    
    scheduler_episode_rewards = [0.0] 


    var=3
    episode_step = 0
    train_step = 0
    scheduler_train_step = 0
    t_start = time.time()
    simulationenv.reset()
    print('Starting iterations...')
    while True:
        simulationenv.run()
        simulationenv.stop_flag = simulationenv.Is_done()  

        ########################################### user trun ###########################################################################
        # 判断是否执行useragent的动作
        for i, user in enumerate(simulationenv.Users): 
            if simulationenv.stopflag is False:
                if (simulationenv.sys_time-user.activeTime)%update_time==0 and simulationenv.sys_time>0:
                    if user.active==True:
                        obs = simulationenv.get_all_user_state()
                        obs_single=simulationenv.get_single_user_state(user.id)

                        userenv.obs_list[user.id].append(obs)
                        userenv.single_obs_list[user.id].append(obs_single)

                        # print("user-"+str(i)+"--state")
                        # print(obs_single)
                        # print("user-"+str(i)+"--all--state")
                        # print(obs)
                        action_user=trainers[i].actor.choose_action(obs_single)
                       # print("user-action:"+str(action_user)+"time:"+str(simulationenv.sys_time))
                        # userenv.action_list[user.id].append(action_user)
                        if len(userenv.action_list) < user.id + 1:
                            for i in range(len(userenv.action_list), user.id  + 1):
                                userenv.action_list.append([])
                        userenv.action_list[user.id].append(action_user)

                        new_user_action=simulationenv.set_user_action(action_user)
                        # new_user_action=[15,1]
                        print("user-"+str(user.id)+"-action:"+str(new_user_action)+"time:"+str(simulationenv.sys_time))
                        #user environment step
                        rew, done = simulationenv.user_step(new_user_action,user.id)
                        if len(userenv.rew_list) < user.id + 1:
                            for i in range(len(userenv.rew_list), user.id  + 1):
                                userenv.rew_list.append([])
                        userenv.rew_list[user.id].append(rew)
                        # print("user-"+str(user.id)+"-reward:"+str(rew)+"time:"+str(simulationenv.sys_time))
                        

                            
                        # update all trainers, if not in display or benchmark mode
                        loss = None
                        if len(userenv.obs_list[user.id])>1:
                            # increment global step counter
                            train_step += 1  

                            obs=userenv.obs_list[user.id][-2]
                            new_obs=userenv.obs_list[user.id][-1]

                            single_obs=userenv.single_obs_list[user.id][-2]
                            new_single_obs=userenv.single_obs_list[user.id][-1]

                            rew=userenv.rew_list[user.id][-2]

                            td_error = trainers[i].critic.learn(obs, rew, new_obs)  # gradient = grad[r + gamma * V(s_) - V(s)]
                            trainers[i].actor.learn(single_obs, action_user, td_error)     # true_gradient = grad[logPi(s,a) * td_error]
                            # print("user-"+str(user.id)+"learn")
                            episode_rewards[-1] += rew
                            agent_rewards[user.id][-1] += rew

        ######################################## scheduler turn #########################################################################
        if  simulationenv.stop_flag is False:
            for gpu in simulationenv.BaseStation.gpu_cluster:
                if gpu.busy_flag is False and gpu.batch_flag is False:
                    state=simulationenv.get_scheduler_state()
                    # print("state")
                    # print(state)
                    schedulerenv.state_list.append(state)
                    action=schedulerTrainer.trainer.choose_action(state)
                    action = np.clip(np.random.normal(action, var), -1, 1)
                    
                    new_action=simulationenv.set_scheduler_action_for_ddpg(action)
                    # new_action=[0,4]
                    # print("scheduler-action:"+str(new_action)+"time:"+str(simulationenv.sys_time))
                    simulationenv.wait_todo.append([gpu.id,new_action])
                    simulationenv.proceed(True,gpu.id,new_action)     
                    # else:
                    #     reward=-10000
                    #     scheduler_episode_rewards[-1] += reward
                    #     # agent.replay_buffer.push((state, action, reward, np.float(done)))
                    #     break

            for gpu in simulationenv.BaseStation.gpu_cluster:
                reward,flag = simulationenv.scheduler_step(gpu)
                if flag:
                    # print("reward:"+str(reward))
                    new_action=gpu.action
                    simulationenv.wait_todo.remove([gpu.id,new_action])
                    if gpu.batch.real_size==pow(2,gpu.batch.size_log):
                        schedulerenv.action_list.append(new_action)
                        scheduler_episode_rewards[-1] += reward
                        schedulerTrainer.trainer.store_transition(state, action, reward )
            
            if schedulerTrainer.trainer.pointer > arglist.batch_size:
                scheduler_train_step += 1
                var *= .9995    # decay the action randomness
                schedulerTrainer.trainer.learn()

       
       

        #teminal是一次episode结束
        # terminal = (episode_step >= arglist.max_episode_len)
        terminal=simulationenv.Is_done()

        
        # done = all(done_n)
        
        simulationenv.sys_time += 1  

        episode_step += 1
        if  terminal:

            ave_r=simulationenv.get_whole_avr_qos()
            user_ave_qos=simulationenv.get_user_avr_qos()
            fw = open('query_qos.txt', 'a')
            fw.write( str(int(scheduler_episode_rewards[-1]))+' '+str(ave_r)+' '+str(user_ave_qos[0])+' '+str(user_ave_qos[1])+' '+str(user_ave_qos[2])+'\n')
            fw.close()
            print('Episode:', len(episode_rewards), 'train step',scheduler_train_step,' Reward: %i' % int(scheduler_episode_rewards[-1]),'ave_r:', ave_r, '|', user_ave_qos[0], '|', user_ave_qos[1], '|', user_ave_qos[2],)
            
            print("steps: {}, episodes: {},  episode reward: {}, agent episode reward: {}, time: {}".format(
                train_step, len(episode_rewards)-1, episode_rewards[-1],
                [rew[-1] for rew in agent_rewards], round(time.time()-t_start, 3)))
            simulationenv.reset()
            episode_step = 0
            episode_rewards.append(0)
            scheduler_episode_rewards.append(0)       
            for a in agent_rewards:
                a.append(0)

        # save scheduler model, display training output
        if terminal and (len(scheduler_episode_rewards) % arglist.save_rate == 0):
            saver.save(sess, arglist.save_dir, global_step= len(scheduler_episode_rewards)-1)
             # print statement depends on whether or not there are adversaries

            print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                     train_step, len(episode_rewards)-1, np.mean(episode_rewards[-arglist.save_rate:]),
                     [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
            print("steps: {}, episodes: {}, mean scheduler episode reward: {}, time: {}".format(
                     scheduler_train_step,len(episode_rewards)-1,np.mean(scheduler_episode_rewards[-arglist.save_rate:]),
                     round(time.time()-t_start, 3)))
            t_start = time.time()
        #     # Keep track of final episode reward
        #     final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
        #     for rew in agent_rewards:
        #         final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
            
        if len(episode_rewards) > arglist.num_episodes:
            break
        # saves final episode reward for plotting training curve later
        # if len(episode_rewards) > arglist.num_episodes:
        #     rew_file_name = arglist.plots_dir + arglist.exp_name + "_rewards.pkl"
        #     with open(rew_file_name, 'wb') as fp:
        #         pickle.dump(final_ep_rewards, fp)
        #     agrew_file_name = arglist.plots_dir + arglist.exp_name + "_agrewards.pkl"
        #     with open(agrew_file_name, 'wb') as fp:
        #         pickle.dump(final_ep_ag_rewards, fp)
        #     print('...Finished total of {} episodes.'.format(len(episode_rewards)))
        #     break

        

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
    
