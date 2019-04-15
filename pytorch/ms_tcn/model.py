#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class MultiStageModel(nn.Module):
    # num_stages = 4  num_layers = 10  num_f_maps = 64  4个stage，一个模块10层，没层都是64个filter
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])
        #   copy.deepcopy 深拷贝 拷贝对象及其子对象

    def forward(self, x):
        out = self.stage1(x)
        outputs = out.unsqueeze(0)   # (1,bz,C_out,L_out)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))  # （ bz,C_out,L_out） dim是要计算的维度
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            #  将每个stage的output做并列拼接
            #  最后维度（4,bz,num_classes,max）
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)  # kernel_size = 1 用1x1conv 降维，从2048降到64
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        # 一维卷积层，输入的尺度是(N, C_in,L)，输出尺度（ N,C_out,L_out）  加padding后 L_out= L
        #  kernel_size=3 dilation=2 ** i 随层数指数递增 1~512, 感受野计算：lk = l(k-1) + 2*dilation, 最后一层每个filter的感受野2047
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)    # 尝试 x+out后再加bn和relu
        return x + out


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                #  batch_input_tensor:（bz, 2048, max), batch_target_tensor: (bz, max), mask: (bz, num_class, max)
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)   # predictions最后一层的输出维度(4,bz,num_classes,max)

                loss = 0
                for p in predictions:
                    #  target:将样本和标签转换数据维度，分别转成二维和一维，最后一维都是类别数
                    #  p.transpose(2, 1)交换维度 （bz,max,num_classes）
                    #  contiguous:返回一个内存连续的有相同数据的tensor，如果原tensor内存连续则返回原tensor
                    #  view():返回一个有相同数据但大小不同的tensor view(-1, self.num_classes)：将转成（bz*max, num_classes）
                    #  batch_target.view(-1):转成(bz*max)
                    #  nn.CrossEntropyLoss(ignore_index=-100):   Target: (N) N是mini-batch的大小，0 <= targets[i] <= C-1
                    #  loss(x,class)=−logexp(x[class])∑jexp(x[j])) =−x[class]+log(∑jexp(x[j]))  Input: (N,C) C 是类别的数量



                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    #  torch.clamp(input, min, max, out=None) → Tensor将输入input张量每个元素的夹紧到区间 [min,max]
                    #  nn.MSELoss(reduction='none')  F.log_softmax(p[:, :, 1:], dim=1):对所有类别求log(softmax)
                    #  dim类别维度 (int): A dimension along which log_softmax will be computed.
                    #  detach():返回一个新的 从当前图中分离的 Variable,被detach 的Variable 指向同一个tensor
                    #  对p中的向量，分别从max维度的：1~max帧和从0~max-1帧划分，错位做均方误差  x，y维度:(bz,max-1)
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                         F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
