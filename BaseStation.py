#-*- coding: UTF-8 -*-

class Histogram(object):
    def __init__(self):
        self.query_bins = []
        self.query_bins_num=[]
        self.deltaT = 200

    def add_in_gram(self, query):
        binID = int((query.geneTime + query.softddl) / self.deltaT)
        query.bin_id=binID
        if len(self.query_bins) < binID + 1:
            for i in range(len(self.query_bins), binID + 1):
                self.query_bins.append([])
                self.query_bins_num.append(0)
        self.query_bins[binID].append(query)
        self.query_bins_num[binID]+=1

    def compute(self):
        num = 0
        for temp in self.query_bins:
            if len(temp):
                num += len(temp)
        return num


class Batch(object):
    def __init__(self, size_log, model, gpu_type):
        self.throughputList = \
        [[[25.6, 35.7, 51.9, 66.1, 73.7, 84.7, 91.2],
          [9.4, 9.4, 12.3, 14.7, 16.0, 16.8, 18.0]],
        [[23.8, 35.7, 63.5, 90.9, 109.6, 131.1, 141.9],
          [14.5, 18.9, 24.2, 28.4, 31.5, 33.9, 34.6]]]
        self.size_log = size_log  # 0 - 6
        self.real_size = 0
        self.model_type = model
        self.gpu_type = gpu_type
        self.throughput = self.throughputList[self.gpu_type][self.model_type][self.size_log]
        self.batch_query = []


class GPU(object):
    def __init__(self, id):
        self.id = id
        self.busy_flag = False  # gpu是否在忙的标记
        self.batch_flag = False  # gpu是否决定下个batch参数的标记
        self.now_endTime = 0  # 处理当前正在处理的Batch的剩余时间
        self.batch = None
        self.cur_serve_user=None
        self.action=None

    def update(self, cur_time):
        if cur_time == self.now_endTime and self.busy_flag is True:
            self.busy_flag = False
            self.now_endTime = 0


class BaseStation(object):
    def __init__(self):
        self.gpu_num = 1
        self.gpu_cluster = []
        self.Histogram = Histogram()
        for i in range(0, self.gpu_num):
            self.gpu_cluster.append(GPU(i))
