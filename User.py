#-*- coding: UTF-8 -*-

from BaseStation import Histogram
ObT = 1e-3  # 系统最小观测时间1ms
update_time = int(10 / ObT)  # 用户更新参数时间为10s


class Query(object):
    def __init__(self, user_id, f, d, gene_time, arrival_time, size, query_id):

        self.Qd = {0.2: 0.32, 0.4: 0.77, 0.6: 0.90, 0.8: 0.96, 1: 1}  # QoS与图像压缩率的函数
        # self.Qd = {0.2: 0.1, 0.4: 0.30, 0.6: 0.50, 0.8: 0.70, 1: 1}  # QoS与图像压缩率的函数
        self.Qm = {0: 0.6, 1: 1}

        self.id = query_id
        self.user_id = user_id
        self.bin_id=0
        self.frame_rate = f
        self.denseness_rate = d
        self.geneTime = gene_time
        self.arrivalTime = arrival_time
        self.size = size
        self.softddl = 1000
        self.hardddl = 2000

        self.model = -1
        self.QoS = -1
        self.end_time = -1
        self.end_flag = False

    def qos_compute(self):
        if self.end_flag is True:
            if self.end_time - self.geneTime <= self.softddl:
                self.QoS = self.Qd[self.denseness_rate] * self.Qm[self.model]
            elif self.end_time - self.geneTime > self.hardddl:
                self.QoS = 0
            else:
                self.QoS = self.Qd[self.denseness_rate] * self.Qm[self.model] * \
                           (1 - (self.end_time - self.geneTime - self.softddl) / (self.hardddl - self.softddl))


class User(object):
    def __init__(self, cur_time, id, active_flag, frame_rate=15, denseness_rate=1, upload_band=20000):
        self.id = id
        self.frame_rate = frame_rate
        self.denseness_rate = denseness_rate
        self.upload_band = upload_band  # 上行带宽，单位kbps
        self.cur_query_list = []  # 当前10s内用户的请求列表
        self.query_list = []  # 当前用户产生的query列表，列表中套列表，元素是每10s产生的query列表
        self.query_num = 0
        self.Histogram = Histogram()
        self.active=active_flag
        self.activeTime=-1
        self.live_time=-1
        self.last_query_arrival=False
        # # 初始化时第一次生成请求列表
        # if self.active:
        #     self.generate_query(cur_time)
        
    def reset(self):
        self.frame_rate = 15
        self.denseness_rate = 1
        self.upload_band = 20000  # 上行带宽，单位kbps
        self.cur_query_list = []  # 当前10s内用户的请求列表
        self.query_list = []  # 当前用户产生的query列表，列表中套列表，元素是每10s产生的query列表
        self.query_num = 0
        self.Histogram = Histogram()
        self.active=False
        self.activeTime=-1
        self.live_time=-1
        self.last_query_arrival=False
        self.live_duration=0

    def generate_query(self, cur_time):
        # self.cur_query_list = []
        for i in range(int(update_time * self.frame_rate *ObT)):
            # 计算大小（是压缩率的函数）单位KB
            size = 150 * self.denseness_rate
            # 计算生成时间
            gene_time = int(cur_time + update_time / ((update_time * ObT) * self.frame_rate) * i)
            # 计算到达时间
            arrival_time = int(gene_time + size / (self.upload_band / 8 * ObT))

            query = Query(self.id, self.frame_rate, self.denseness_rate, gene_time, arrival_time, size, self.query_num)
            self.query_num += 1
            self.cur_query_list.append(query)
        self.query_list.append(self.cur_query_list)

    def update_parameters(self, cur_time,frame_rate,denseness_rate):

        # 根据平均qos调整帧率、压缩率
        self.frame_rate = frame_rate
        self.denseness_rate = denseness_rate
        
        # 根据调整后的帧率，压缩率产生新的10s的请求
        if self.active:
            self.generate_query(cur_time)
