import os
os.environ['CUDA_DEVICE_ORDER'] = '0'
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import transforms

import numpy as np
from models.i3d import InceptionI3d
import mmcv
from utils.videotransforms import CenterCrop
import scipy.misc
from moviepy.editor import VideoFileClip


def i3d_feature(frame_dir, save_dir='./'):
    for file in os.listdir(frame_dir):
        if os.path.isdir(file):
            sub_dir = frame_dir + './' + file
            input_shape = (224, 224)
            frames = [scipy.misc.imresize(CenterCrop(x), input_shape) for x in os.listdir(sub_dir)]
            i3d = InceptionI3d(400, in_channels=3)
            i3d.replace_logits(157)
            i3d.load_state_dict(torch.load('../checkpoints/rgb_imagenet.pt'))
            i3d.cuda()
            features = []
            for frame in frames:
                ip = Variable(torch.from_numpy(frame.numpy()).cuda(), volatile=True)
                features.append(i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
            feat_filepath = os.path.join(save_dir, file + '.npy')
            with open(feat_filepath, 'wb') as f:
                np.save(f, np.concatenate(features, axis=0))



    print(frames)


i3d_feature('/Users/user/Desktop/openSourceML/myBornflyCV/pytorch/learn_mmcv/PETS.avi')
# # Iterate over data.
# for data in dataloaders[phase]:
#     # get the inputs
#     inputs, labels, name = data
#     if os.path.exists(os.path.join(save_dir, name[0] + '.npy')):
#         continue
#
#     b, c, t, h, w = inputs.shape
#     if t > 1600:
#         features = []
#         for start in range(1, t - 56, 1600):
#             end = min(t - 1, start + 1600 + 56)
#             start = max(1, start - 48)
#             ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda(), volatile=True)
#             features.append(i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
#         np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
#     else:
#         # wrap them in Variable
#         inputs = Variable(inputs.cuda(), volatile=True)
#         features = i3d.extract_features(inputs)
#         np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())