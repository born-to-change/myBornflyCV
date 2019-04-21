import os
os.environ['CUDA_DEVICE_ORDER'] = '4,5,6,7'
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import torchvision
from torchvision import transforms

import numpy as np
from models.i3d import InceptionI3d
import mmcv
from utils.videotransforms import CenterCrop
import scipy.misc
# from moviepy.editor import VideoFileClip

def crop_center(im):
    """
    Crops the center out of an image.
    Args:
        im (numpy.ndarray): Input image to crop.
    Returns:
        numpy.ndarray, the cropped image.
    """

    h, w = im.shape[0], im.shape[1]

    if h < w:
        return im[0:h, int((w - h) / 2):int((w - h) / 2) + h, :]
    else:
        return im[int((h - w) / 2):int((h - w) / 2) + w, 0:w, :]


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(frame_dir, file, start=0):
    frames = []
    num = len(os.listdir(frame_dir + '/' + file))
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(frame_dir, file, str(i) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)

    imgs = np.asarray(frames, dtype=np.float32)

    return imgs



def i3d_feature(frame_dir, save_dir):
    for file in os.listdir(frame_dir):
        if os.path.isdir(frame_dir+'/'+file):
            sub_dir = frame_dir + '/' + file
        else:
            sub_dir = frame_dir
        input_shape = (224, 224)

        i3d = InceptionI3d(400, in_channels=3)
        #i3d.replace_logits(157)
        i3d.load_state_dict(torch.load('../checkpoints/rgb_imagenet.pt'))
        i3d = torch.nn.DataParallel(i3d, device_ids=[0,1,2,4])
        #i3d.cuda()
        features = []

        # b,c,t,h,w = inputs.shape
        # if t > 1600:
        #     features = []
        #     for start in range(1, t - 56, 1600):
        #         end = min(t - 1, start + 1600 + 56)
        #         start = max(1, start - 48)
        #         ip = Variable(torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda(), volatile=True)
        #         features.append(i3d.module.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
        #     np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
        # else:
        #     # wrap them in Variable
        #     inputs = Variable(inputs.cuda(), volatile=True)
        #     features = i3d.extract_features(inputs)
        #     np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())

        # for x in os.listdir(sub_dir):
        #     frame = mmcv.imread(sub_dir + '/' + x)
        #     frame = scipy.misc.imresize(crop_center(frame).astype(np.float32), input_shape)

        frames = load_rgb_frames(frame_dir, file)

        ip = Variable(torch.from_numpy(frames), volatile=True)

        features.append(i3d.module.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())
        feat_filepath = os.path.join(save_dir, file + '.npy')
        with open(feat_filepath, 'wb') as f:
            np.save(f, np.concatenate(features, axis=0))




i3d_feature('/disk2/lzq/data/merlshop/frames', '/disk2/lzq/data/merlshop/features')
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