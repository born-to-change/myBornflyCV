#!/usr/bin/python2.7

import torch
import numpy as np
import random


class BatchGenerator(object,):
    def __init__(self, num_classes, batch_size, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if (len(self.list_of_examples)-self.index) > self.batch_size:
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)    # 读取训练集的视频个数

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')  # 读取特征
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]       # 每一帧的label
            #  特征行数和groundtruth取最小的一个，作为训练样本对数
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]    # 将单词转化为数字
            batch_input .append(features[:, ::self.sample_rate])    # feature维度（2048, x） 隔sample_rate个样本采样（大于1时跳帧采样）
            batch_target.append(classes[::self.sample_rate])     # 生成每个特征的标签  （frames，）

        # 对batch_target每个元素求长度  python2 中返回一个list，Python3返回迭代器
        length_of_sequences = map(len, batch_target)
        # （bz,2048,max) max:bz中帧数最多的视频样本
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        # (bz, max) 且元素全部赋初始值为-100
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        # (bz, num_class, max)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            # numpy转换为torch的tensor，且从二维转换到三维固定维度，加上batch_size
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            # numpy转换为torch的tensor，且从一维转换到二维固定维度，加上batch_size
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            # 生成掩膜，为了保存每个视频样本的帧数,在总帧数为max中 用mask为1区分不同样本的帧数  (bz, num_class, max)就是一个one-hot标签
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        #  batch_input_tensor:（bz,2048,max), batch_target_tensor:(bz, max), mask:(bz, num_class, max)
        return batch_input_tensor, batch_target_tensor, mask
