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
class SimulationEnv(object):
    def __init__(self, user_env, scheduler_env, arglist):
        self.user_env=user_env
        self.scheduler_env=scheduler_env
        self.stopflag = False
        self.realstopflag = False
        self.all_user_queries_buffer = []  # 记录所有用户每10s请求的缓存区
        self.Users = []  # 用户列表
        self.User_num = arglist.user_num  # 最大用户数量
        self.user_active_flag=np.zeros(self.User_num,np.float32)
        self.Cur_user_num=0 #当前用户数量
        self.BaseStation = BaseStation()  # 基站
        self.sys_time = 0  # 系统时间，单位ObT
        self.last_time=0
        self.Qos_memory = []  # 保存请求的用户以及QoS
        self.Qos_memory_index=0
        self.queries_num = 0  # 系统在时间段内处理的请求数量
        self.queries_num_for_user=[]
        self.user_arrival_time=[]
        self.get_user_arrival_time()
        self.deltaT = 200
        # self.s_dim=(N_post+N_pre)*self.User_num+self.User_num*2
        # self.a_dim=14
        self.s_dim=scheduler_env.s_dim
        self.a_dim=scheduler_env.a_dim
        self.fairness_count=-1
        self.drop_num=0
        self.wait_todo=[]
        self.fairness_index_list=[]
        self.init()
        self.live_flag=np.zeros(self.User_num,np.int)
        self.update_live_flag()
        self.Jian_index=0

    def init(self):
        for i in range(0, self.User_num):
            self.Users.append(User(0, i,False, frame_rate=15, denseness_rate=1))
            self.all_user_queries_buffer += self.Users[i].cur_query_list
        self.all_user_queries_buffer.sort(key=lambda x: x.arrivalTime, reverse=False)
        self.queries_num += len(self.all_user_queries_buffer)

            
    def get_user_arrival_time(self):
        fw=open("user_arrival_time.txt")
        for line in fw.readlines():
            lineArr = line.strip().split(' ')
            user=int(lineArr[0])
            arrival_time = int(lineArr[1])
            live_duration=int(lineArr[2])
            self.user_arrival_time.append([user,arrival_time,live_duration])
    def seed(self,seed):
        np.random.seed(seed)
    def reset(self):
        self.user_env.reset()
        self.scheduler_env.reset()
        self.stopflag = False
        self.realstopflag = False
        self.all_user_queries_buffer = []
        self.queries_num = 0 
        self.Users = []  # 用户列表
        self.user_active_flag=np.zeros(self.User_num,np.float32)
        self.Cur_user_num=0 #当前用户数量
        self.sys_time = 0  # 系统时间，单位ObT
        self.last_time=0
        self.Qos_memory = []  # 保存请求的用户以及QoS
        self.Qos_memory_index=0
        self.Qos_memory_for_action=[]
        self.BaseStation = BaseStation()
        self.queries_num_for_user=[]
        self.user_arrival_time=[]
        self.get_user_arrival_time()
        self.fairness_count=-1
        self.drop_num=0
        self.wait_todo=[]
        self.fairness_index_list=[]
        self.init()
        self.live_flag=np.zeros(self.User_num,np.int)
        self.update_live_flag()
        self.Jian_index=0


    def get_idle_gpu(self):
        for gpu in self.BaseStation.gpu_cluster:
            if gpu.busy_flag is False:
                return gpu
            else:
                return None


    def run(self):
        
        #判断是否有新用户到达
        if self.stopflag is False:
            while len(self.user_arrival_time)>0:
                if self.sys_time==self.user_arrival_time[0][1]:
                    if self.Cur_user_num<self.User_num:
                        print("user"+str(self.user_arrival_time[0][0])+"come"+' '+"systime:"+str(self.sys_time))
                        self.Users[self.user_arrival_time[0][0]].active=True
                        self.Users[self.user_arrival_time[0][0]].activeTime=self.sys_time
                        self.Users[self.user_arrival_time[0][0]].live_time=0
                        self.Users[self.user_arrival_time[0][0]].generate_query(self.sys_time)
                        self.Users[self.user_arrival_time[0][0]].live_duration=self.user_arrival_time[0][2]
                        self.Cur_user_num+=1
                        if self.Cur_user_num>1:
                            self.fairness_count=0
                        
                        if self.Cur_user_num>1:
                            self.update_live_flag()
                            self.fairness_index_list.append([len(self.Qos_memory),self.live_flag])
                    else:
                        print("user's num is over capacity!")
                    del self.user_arrival_time[0]
                else:
                    break
                                
         # 更新所有gpu的状态
        for gpu in self.BaseStation.gpu_cluster:
            gpu.update(self.sys_time)

       


        #为每一个用户维护一个histogram
        cur_bin=int(self.sys_time/self.deltaT)
        for i in range(len(self.Users)):
            if len(self.Users[i].cur_query_list)>0:
                self.Users[i].cur_query_list.sort(key=lambda x: x.arrivalTime,reverse=False)
                while len(self.Users[i].cur_query_list):
                    if self.Users[i].cur_query_list[0].arrivalTime==self.sys_time:
                        self.Users[i].Histogram.add_in_gram(self.Users[i].cur_query_list.pop(0))
                    else:
                        break
        
         #在N_pre之前的就删除
        for user in (self.Users):
            if len(user.Histogram.query_bins)>0:
                for j in range(0,min(cur_bin-N_pre,len(user.Histogram.query_bins))):
                    # print("1----:"+str(j)+"----"+str(cur_bin-N_pre)+"----:"+str(len(user.Histogram.query_bins[j])))
                    while len(user.Histogram.query_bins[j]):
                        
                        query=user.Histogram.query_bins[j].pop(0)
                        
                        self.drop_num+=1
                        user.Histogram.query_bins_num[j]-=1
                        self.Qos_memory.append([user.id, query.QoS+1, query.id])
                        # print("drop"+"--------"+"self.Qos_memory:"+str(query.user_id)+"num:"+str(query.id))
                        if len(self.user_env.last_duration_qos) < user.id + 1:
                            for i in range(len(self.user_env.last_duration_qos), user.id  + 1):
                                self.user_env.last_duration_qos.append([])
                        self.user_env.last_duration_qos[user.id].append(query.QoS+1)
                        # self.user_env.last_duration_qos[query.user_id].append(query.QoS)
                        if len(self.queries_num_for_user) < user.id + 1:
                            for i in range(len(self.queries_num_for_user), user.id  + 1):
                                self.queries_num_for_user.append(0)
                        self.queries_num_for_user[query.user_id]+=1
                        # if (query.user_id==0):
                        #     print("drop"+"--------"+"self.queries_num_for_user:"+str(query.user_id)+"num:"+str(query.id))
                        # print("2----:"+str(len(user.Histogram.query_bins[j])))

          #判断用户是否leave
        
        for user in self.Users:
            if user.active ==False and  user.query_num>0  :
                if user.query_num==self.queries_num_for_user[user.id] :
                    print("user"+str(user.id)+"leave"+"cur_bin:"+str(cur_bin)+"time:"+str(self.sys_time))
                    self.Cur_user_num-=1
                    user.reset()
                    if self.Cur_user_num>1 or self.Cur_user_num==1:
                        self.update_live_flag()
                        self.fairness_index_list.append([len(self.Qos_memory),self.live_flag])
                    if self.Cur_user_num>1:
                        self.fairness_count=0

        # #判断用户的最后一个query是否到
        # for user in self.Users:
        #     if user.active==False and user.query_num>0:
        #         if user.Histogram.compute() +self.queries_num_for_user[user.id]==user.query_num:
        #             user.last_query_arrival=True
        #             print("user"+str(user.id)+"last query arrival"+"-----:"+str(user.query_num))
        #             # for item in self.wait_todo:
        #             #     if item[1][self.a_dim+user.id]>0:
        #             #         self.wait_todo.remove(item)
        #             #         self.proceed(False,item[0],item[1])
        #             if len(self.wait_todo)>0:
        #                 for item in self.wait_todo:
        #                     self.wait_todo.remove(item)
        #                     self.proceed(False,item[0],item[1])
                    
        #             gpu=self.get_idle_gpu()
        #             if gpu is not None:
        #                 for item in self.wait_todo:
        #                     if item[0]==gpu.id:
        #                         self.wait_todo.remove(item)
        #                         self.proceed(False,item[0],item[1])
        #                 print("user"+str(user.id)+"Histogram.compute():"+str(user.Histogram.compute())+"time:"+str(self.sys_time))
        #                 action=[0,int(log(user.Histogram.compute(),2))+1]
                        
        #                 print("run--action:")
        #                 print(action)
                    
        #                 # print(True)
        #                 self.wait_todo.append([gpu.id,action])
        #                 self.proceed(True,gpu.id,action)
        #                 gpu.batch.real_size=user.Histogram.compute()
                       
                        

                    # else :
                    #     # print(False)
                    #     break
                        
                    # break
       
        #判断用户是否deactivate
        for i in range(len(self.Users)):
            if self.Users[i].active :
                self.Users[i].live_time+=1
                if self.Users[i].live_time==self.Users[i].live_duration:
                    print("user"+str(self.Users[i].id)+"deactivate")
                    # print("user"+str(self.Users[i].id)+"query num:"+str(self.Users[i].query_num))
                    self.Users[i].active =False

       

        if self.Cur_user_num<=1:
            self.fairness_count=-1
        # 当用户更新参数后，将请求放入系统请求列表
        ######   放进userenv   #######
        # if self.stopflag is False:
        #     for user in self.Users:
        #         if user.active and (self.sys_time-user.activeTime)>0 and (self.sys_time-user.activeTime)%update_time==0:
        #             user.update_parameters(self.sys_time)

       
    
    def update_live_flag(self):
        live_flag=np.zeros(self.User_num,np.int)
        for i in range(0,self.User_num):
            if self.Users[i].active==False and self.Users[i].query_num==0:
                live_flag[i]=0
            else:
                live_flag[i]=1
        self.live_flag=live_flag

    def Is_done(self):
        if self.Cur_user_num==0:
            return True
        else:
            return False

    def bound(self,input,x_min,x_max,y_min,y_max):
        if np.isnan(input):
            print("input is nan")
        result=y_min+(y_max-y_min)/(x_max-x_min)*(input-x_min)
        return (result)

    ######################## for scheduler #########################

    def get_scheduler_state(self):
        s=np.zeros(self.s_dim,np.float32)
        #1、直方图
        # max_num=self.User_num*15/(1000/self.deltaT)
        cur_bin=int(self.sys_time/self.deltaT)
        for i in range(len(self.Users)):
            s1=np.zeros(N_post+N_pre,np.float32)
            if len(self.Users[i].Histogram.query_bins)>0 :
                # print("user.query_num:"+str(self.Users[i].query_num))
                # if self.sys_time>20000:
                #     print(self.queries_num_for_user[i])
                # print("cur_bin:"+str(cur_bin))
                # print(len(self.Users[i].Histogram.query_bins_num))
                # print(self.Users[i].Histogram.query_bins_num)

                if cur_bin<5:
                    s1[(N_pre-cur_bin):(N_pre-cur_bin+len(self.Users[i].Histogram.query_bins))]=self.Users[i].Histogram.query_bins_num[0:min(cur_bin+N_post,len(self.Users[i].Histogram.query_bins))]
                else :
                    s1[0:(len(self.Users[i].Histogram.query_bins)-cur_bin+N_pre)]= self.Users[i].Histogram.query_bins_num[(cur_bin-N_pre):min(cur_bin+N_post,len(self.Users[i].Histogram.query_bins))]
            # s1[:]=[x*10 for x in s1]
            s[(self.Users[i].id*(N_post+N_pre)):((self.Users[i].id+1)*(N_post+N_pre))]=s1 

        # 2、指示
       
        for i in range(len(self.Users)):
            if self.Users[i].active :
                s[self.User_num*(N_post+N_pre)+i]=1
            else:
                s[self.User_num*(N_post+N_pre)+i]=0
        # for i in range(len(self.Users)):
        #     if self.Is_user_leave_for_the_action(self.Users[i]) :
        #         s[self.User_num*(N_post+N_pre)+i]=0
        #     else:
        #         s[self.User_num*(N_post+N_pre)+i]=1
        # if self.Jian_index>=0 and self.Jian_index<1/3:
        #     s[self.User_num*(N_post+N_pre)+3]=1
        #     s[self.User_num*(N_post+N_pre)+4]=0
        #     s[self.User_num*(N_post+N_pre)+5]=0
        # elif self.Jian_index>=1/3 and self.Jian_index<2/3:
        #     s[self.User_num*(N_post+N_pre)+3]=0
        #     s[self.User_num*(N_post+N_pre)+4]=1
        #     s[self.User_num*(N_post+N_pre)+5]=0
        # else:
        #     s[self.User_num*(N_post+N_pre)+3]=0
        #     s[self.User_num*(N_post+N_pre)+4]=0
        #     s[self.User_num*(N_post+N_pre)+5]=1

        s[:]=[x*1 for x in s]
        return s


    def set_scheduler_action(self,a):
        if a>=0 and a<6:
            return [0,a]
        elif a<12:
            return [1,a-6]

    def set_scheduler_action_for_ddpg(self,a):
        new_a0=self.bound(a[0],-1,1,0,2)
        new_a0 = np.clip(int(new_a0),0,1)
        new_a1=self.bound(a[1],-1,1,0,6)
        new_a1 = np.clip(int(new_a1),0,5)
        new_action=[new_a0,new_a1]
        return new_action

    def proceed(self, todo, gpu_id, new_action):
        if todo:
        # 为空闲的gpu创建一个batch
            self.BaseStation.gpu_cluster[gpu_id].batch_flag = True
            self.BaseStation.gpu_cluster[gpu_id].action=new_action
            self.BaseStation.gpu_cluster[gpu_id].batch = Batch(new_action[1], new_action[0], gpu_type=1)
            self.BaseStation.gpu_cluster[gpu_id].batch.real_size=pow(2, new_action[1])
        else:
            self.BaseStation.gpu_cluster[gpu_id].batch_flag = False
            self.BaseStation.gpu_cluster[gpu_id].action=[]
            self.BaseStation.gpu_cluster[gpu_id].batch = []

    def inter_step(self,gpu,q_num):
        waitting_list=[]
        for user in self.Users:
            for bin in user.Histogram.query_bins:
                if len(bin)>0:
                    for query in bin:
                        waitting_list.append(query)
        waitting_list.sort(key=lambda x: x.arrivalTime, reverse=False)
        # for query in waitting_list:
        #     print(str(query.user_id)+"+"+str(query.id))
        for i in range(q_num):
            query=waitting_list.pop(0)
            # print(str(query.user_id)+"+"+str(query.id))
            # print("query-bin-id:"+str(query.bin_id))
            gpu.batch.batch_query.append(query)
            self.Users[query.user_id].Histogram.query_bins_num[query.bin_id]-=1
            self.Users[query.user_id].Histogram.query_bins[query.bin_id].remove(query)

    def inter_step_old(self,gpu,batch_part):
        waitting_list=[]
        for user in self.Users:
            waitting_list+=user.Histogram.query_bins
        for item in (waitting_list):
            item.sort(key=lambda x: x.arrivalTime, reverse=False)
        batch_index = 0
        bin_index=0
        # print("waitting_list")
        # for item in waitting_list:
        #     print(item)
        while True:
            for bin in waitting_list:
                bin_index+=1
                if len(bin) > 0:
                    while len(bin) > 0:
                        batch_index += 1
                        query=bin.pop(0)
                        gpu.batch.batch_query.append(query)
                        self.Users[query.user_id].Histogram.query_bins_num[bin_index-1]-=1
                        if batch_index == batch_part:
                            break
                    if batch_index == batch_part:
                        break
                else:
                    continue
                
            if batch_index == batch_part:
                break
        # while True:
        #     for bin in user.Histogram.query_bins:
        #         bin_index+=1
        #         if len(bin) > 0:
        #             while len(bin) > 0:
        #                 batch_index += 1
        #                 gpu.batch.batch_query.append(bin.pop(0))
        #                 user.Histogram.query_bins_num[bin_index-1]-=1
        #                 if batch_index == batch_part:
        #                     break
        #             if batch_index == batch_part:
        #                 break
        #         else:
        #             continue
                
        #     if batch_index == batch_part:
        #         break
    
    def Is_batch_for_user_satisfied(self,new_action):
        legal=False
        cur_bin=int(self.sys_time/self.deltaT)
        query=[]
        for i in range(len(self.Users)):
            query.append( sum(self.Users[i].Histogram.query_bins_num[0:cur_bin+N_post]) )
        if sum(query)>=pow(2, new_action[1]):
            legal=True
        return legal
 
    def scheduler_step(self,gpu):
        reward=0
        flag=False
        if gpu.batch_flag is True:
            new_action=gpu.action

            if self.Is_batch_for_user_satisfied(new_action):
                
                gpu.busy_flag = True
                gpu.now_endTime = self.sys_time + int(gpu.batch.real_size /
                                                        gpu.batch.throughput / ObT)
                # print("batch satisfied")
#                print("time:"+str(self.sys_time))
#                s=self.get_scheduler_state()
#                print(s)
#                print("preceedtime:"+str(int(gpu.batch.real_size /
#                                                         gpu.batch.throughput / ObT)))
                gpu.batch_flag = False
                
                self.inter_step(gpu,pow(2, new_action[1]))
                
                Qos_memory=[]
                for query in gpu.batch.batch_query:
                    query.end_time = gpu.now_endTime
                    query.model = gpu.batch.model_type
                    query.end_flag = True
                    query.qos_compute()
                    if gpu.batch.real_size==pow(2,gpu.batch.size_log):
                        self.Qos_memory.append([query.user_id, query.QoS, query.id])
                        # print("preceed"+"--------"+"self.Qos_memory:"+str(query.user_id)+"num:"+str(query.id))
                        Qos_memory.append([query.user_id, query.QoS, query.id])
                        # self.user_env.last_duration_qos[query.user_id].append(query.QoS)
                        if len(self.user_env.last_duration_qos) < query.user_id + 1:
                            for i in range(len(self.user_env.last_duration_qos), query.user_id  + 1):
                                self.user_env.last_duration_qos.append([])
                        self.user_env.last_duration_qos[query.user_id].append(query.QoS)

                    
                    if len(self.queries_num_for_user) < query.user_id + 1:
                        for i in range(len(self.queries_num_for_user), query.user_id  + 1):
                            self.queries_num_for_user.append(0)
                    self.queries_num_for_user[query.user_id]+=1
                    # if (query.user_id==0):
                    #     print("preceed"+"--------"+"self.queries_num_for_user:"+str(query.user_id)+"num:"+str(query.id))
                self.Qos_memory_for_action.append(Qos_memory)
                #计算reward
                qos_total=[]
                if len(self.Qos_memory)<=100:
                    for i in range(0,len(self.Qos_memory)):
                        qos_total.append(self.Qos_memory[i][1])
                else:
                    for i in range(len(self.Qos_memory)-100,len(self.Qos_memory)):
                        qos_total.append(self.Qos_memory[i][1])
#                for i in range(self.Qos_memory_index,len(self.Qos_memory)):
#                    qos_total.append(self.Qos_memory[i][1])
#                self.Qos_memory_index=len(self.Qos_memory)
#                reward1=sum(qos_total)
                reward1=sum(qos_total)/len(qos_total)




                user_qos=np.zeros(self.User_num,np.float32)
                for i in range (0,len(self.Qos_memory_for_action)):
                    for item in self.Qos_memory_for_action[i]:
                        user_qos[item[0]]+=item[1]
                user_qos_=[]
                for i in range(len(user_qos)):
                    if self.Users[i].active==False and self.Users[i].query_num==0:
                        continue
                    else:
                        user_qos_.append(user_qos[i]/(self.sys_time-self.last_time))
                user_qos_s=[np.square(x) for x in user_qos_]
                reward2=np.square(sum(user_qos_))/(len(user_qos_)*sum(user_qos_s))
                self.Jian_index=reward2
                reward=reward1
                # reward=exp(-(np.spuare(reward1-1)/(2*1))+exp(-(np.spuare(reward2-1)/(2*1))
                # print("time:"+str(self.sys_time))
                # print("fairnesscount:"+str(self.fairness_count))
#                print("reward1:"+str(reward1))
                # print("reward2:"+str(reward2))
                # print(user_qos_)
#                print("Qos_memory")
#                print(self.Qos_memory)
                # print("Qos_memory_for_action")
                # print(self.Qos_memory_for_action)
                
                flag=True
                return reward, flag
            else:
                return reward,flag
        else:
            return reward,flag



    def get_whole_avr_qos(self):
        qos_total=[]
        
        for i in range(0,len(self.Qos_memory)):
            qos_total.append(self.Qos_memory[i][1])
        ave_r=sum(qos_total)/len(qos_total)
        return ave_r


    ############################# for user #####################
    def get_all_user_state(self):
        # s=[np.zeros(self.user_env.s_dim,np.float32) for user in self.Users]
        s=np.zeros(self.user_env.s_dim,np.float32)
        cur_bin=int(self.sys_time/self.deltaT)
        for i in range(len(self.Users)):
            s1=np.zeros(N_post+N_pre,np.float32)
            if len(self.Users[i].Histogram.query_bins)>0 :
                if cur_bin<5:
                    s1[(N_pre-cur_bin):(N_pre-cur_bin+len(self.Users[i].Histogram.query_bins))]=self.Users[i].Histogram.query_bins_num[0:min(cur_bin+N_post,len(self.Users[i].Histogram.query_bins))]
                else :
                    s1[0:(len(self.Users[i].Histogram.query_bins)-cur_bin+N_pre)]= self.Users[i].Histogram.query_bins_num[(cur_bin-N_pre):min(cur_bin+N_post,len(self.Users[i].Histogram.query_bins))]

            s[(self.Users[i].id*(N_post+N_pre)):((self.Users[i].id+1)*(N_post+N_pre))]=s1 


        s[self.User_num*(N_post+N_pre)]=self.Jian_index

        s2=np.zeros(10,np.float32)
        s2[:]=[0 for x in s2]
        if len(self.scheduler_env.action_list)>=5:
            s2=np.hstack(self.scheduler_env.action_list[ -5:])
        else:
            for  i in range(len(self.scheduler_env.action_list)):
                s2[9-len(self.scheduler_env.action_list)+i:11-len(self.scheduler_env.action_list)+i]=self.scheduler_env.action_list[i]
        # print(self.scheduler_env.action_list)
        # print(s2)
        s[(self.User_num*(N_post+N_pre)+1):(self.User_num*(N_post+N_pre)+11)]= s2

        for i in range(len(self.Users)):
            aver_qos=np.mean(self.user_env.last_duration_qos[i])
            # print("aver_qos:"+str(aver_qos))
            if np.isnan(aver_qos):
                s[self.User_num*(N_post+N_pre)+11+i]=0
            else:
                s[self.User_num*(N_post+N_pre)+11+i]=aver_qos
        next_index=self.User_num*(N_post+N_pre)+13
        
        # if len(self.user_env.action_list)==1:
        #     for i in range(1):
        #         if len(self.user_env.action_list[i])>0:
        #             s[next_index:next_index+2]=self.user_env.action_list[i][-1]
        #         next_index=next_index+2
        # elif len(self.user_env.action_list)==2:
        #     for i in range(2):
        #         if len(self.user_env.action_list[i])>0:
        #             s[next_index:next_index+2]=self.user_env.action_list[i][-1]
        #         next_index=next_index+2
        # else len(self.user_env.action_list)==3:
        #     for i in range(3):
        #         if len(self.user_env.action_list[i])>0:
        #             s[next_index:next_index+2]=self.user_env.action_list[i][-1]
        #         next_index=next_index+2
        return s

    def get_single_user_state(self,user_id):
        s=s=np.zeros(self.user_env.s_single_dim,np.float32)
        cur_bin=int(self.sys_time/self.deltaT)
        s1=np.zeros(N_post+N_pre,np.float32)
        if len(self.Users[user_id].Histogram.query_bins)>0 :
            if cur_bin<5:
                s1[(N_pre-cur_bin):(N_pre-cur_bin+len(self.Users[user_id].Histogram.query_bins))]=self.Users[user_id].Histogram.query_bins_num[0:min(cur_bin+N_post,len(self.Users[user_id].Histogram.query_bins))]
            else :
                s1[0:(len(self.Users[user_id].Histogram.query_bins)-cur_bin+N_pre)]= self.Users[user_id].Histogram.query_bins_num[(cur_bin-N_pre):min(cur_bin+N_post,len(self.Users[user_id].Histogram.query_bins))]

        s[0:(N_post+N_pre)]=s1 

        s[N_post+N_pre]=self.Jian_index

        s2=np.zeros(10,np.float32)
        s2[:]=[0 for x in s2]
        if len(self.scheduler_env.action_list)>=5:
            s2=np.hstack(self.scheduler_env.action_list[ -5:])
        else:
            for  i in range(len(self.scheduler_env.action_list)):
                s2[9-len(self.scheduler_env.action_list)+i:11-len(self.scheduler_env.action_list)+i]=self.scheduler_env.action_list[i]
        s[((N_post+N_pre)+1):((N_post+N_pre)+11)]= s2

        aver_qos=np.mean(self.user_env.last_duration_qos[user_id])
        # print("aver_qos:"+str(self.user_env.last_duration_qos[user_id]))
        if np.isnan(aver_qos):
            s[(N_post+N_pre)+11]=0
        else:
            s[(N_post+N_pre)+11]=aver_qos

        return s

    def set_user_action(self,a):
        if a>=0 and a<5:
            return [10,round((a+1)*0.2,1)]
        elif a<10:
            return [15,round((a-4)*0.2,1)]
       
        elif a<15:
            return [20,round((a-9)*0.2,1)]

    def set_user_action_for_ddpg(self,a):
        new_a0=self.bound(a[0],-1,1,2,5)
        new_a0 = np.clip(int(new_a0),2,4)
        new_a1=self.bound(a[1],-1,1,1,6)
        new_a1 = np.clip(int(new_a1),1,5)
        new_action=[new_a0*5,new_a1*0.2]
        return new_action

    def user_step(self,new_action,user_id):
        self.Users[user_id].update_parameters(self.sys_time,new_action[0],new_action[1])
        reward=0
        rew1=np.mean(self.user_env.last_duration_qos[user_id])
        if np.isnan(rew1):
            rew1=0

        rew2=self.Jian_index
        # reward=2*rew1+rew2
        reward=rew1
        done=self.Users[user_id].active
        return reward, done

          # 当用户更新参数后，将请求放入系统请求列表
        

    def get_user_avr_qos(self):
        user_qos=[]
        for i in range(self.User_num):
            user_qos.append([])
        for i in range (0,len(self.Qos_memory)):
            user_qos[self.Qos_memory[i][0]].append(self.Qos_memory[i][1])
        return sum(user_qos[0])/len(user_qos[0]),sum(user_qos[1])/len(user_qos[1]),sum(user_qos[2])/len(user_qos[2])
    
 
        
if __name__ == "__main__":
    MEC_AR_ENV = Env(30000)
    MEC_AR_ENV.reset()
    ep_reward=0
    while True:
        # print ("systime:"+str(MEC_AR_ENV.sys_time))
        MEC_AR_ENV.run()
        # 如果有空闲的cpu，并且该gpu还未进行决策，则agent进行一次决策，输出action
        for gpu in MEC_AR_ENV.BaseStation.gpu_cluster:
            if gpu.busy_flag is False and gpu.batch_flag is False:
                s=MEC_AR_ENV.get_state()
                # print("s")
                # print(s)
                # if MEC_AR_ENV.sys_time<=10000:
                #     action=[4,0,0,2,1]
                # elif MEC_AR_ENV.sys_time<20000:
                #     action=[8,8,0,4,1]
                # elif MEC_AR_ENV.sys_time<30000:
                #     action=[11,11,10,5,1]
                # else:
                #     action=[0,0,16,4,1]
                # print("a"+str(MEC_AR_ENV.sys_time))
                # print(action)
                action=MEC_AR_ENV.random_action()
                new_action=MEC_AR_ENV.translate_action(action)
                # print("a"+str(MEC_AR_ENV.sys_time))
                # print(action)
                # print(new_action)
                if MEC_AR_ENV.Is_action_legal(new_action)==False:
                    r=-10000
                    ep_reward+=r
                    # print("r:"+str(r))
                else:
                    MEC_AR_ENV.wait_todo.append([gpu.id,new_action])
                    MEC_AR_ENV.proceed(True,gpu.id,new_action)
        for gpu in MEC_AR_ENV.BaseStation.gpu_cluster:
            
            reward,flag=MEC_AR_ENV.step(gpu)
            if flag:
                # print(reward)
                ep_reward+=reward
                action=gpu.action
                MEC_AR_ENV.wait_todo.remove([gpu.id,action])

        MEC_AR_ENV.stopflag=MEC_AR_ENV.Is_done()
        if MEC_AR_ENV.stopflag:
            break
        
        MEC_AR_ENV.sys_time += 1
    print (ep_reward)
    # print(len(MEC_AR_ENV.Qos_memory))
    # print(MEC_AR_ENV.Qos_memory)
