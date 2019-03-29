from scipy.io import loadmat
import cv2
import os
from moviepy.editor import VideoFileClip
import numpy as np


# 0 background
# 1 reach_to_shelf
# 2 retract_from_shelf
# 3 hand_in_shelf
# 4 inspect_product
# 5 inspect_shelf

video_dir = "/disk2/lzq/Videos_MERL_Shopping_Dataset"
label_dir = "/disk2/lzq/data/MERL/Labels_MERL_Shopping_Dataset/"
groundtruth_dir = "/disk2/lzq/data/MERL/groundTruth/"

map_dict = {'0': 'background', '1': 'reach_to_shelf',
            '2': 'retract_from_shelf',
            '3': 'hand_in_shelf',
            '4': 'inspect_product',
            '5': 'inspect_shelf'}

for video in os.listdir(video_dir):
    video_path = video_dir + '/' + video
    clip = VideoFileClip(video_path)
    label_nums = clip.duration * clip.fps

    label_file = video.split("crop")[0] + 'label.mat'

    label_path = label_dir + label_file
    labels = loadmat(label_path)

    groundtruth_file = video.split("c")[0] + 'label.txt'
    list = np.array((labels['tlabs']))  # shape:(5, 1)


    label_dict = {}
    for i, type in enumerate(list):
        for action in type[0]:
            for index in range(action[0], action[1] + 1):
                label_dict[str(index)] = map_dict[str(i + 1)]
        for row in range(label_nums):
            if str(row + 1) not in label_dict.keys():
                label_dict[str(row + 1)] = map_dict['0']

    print(len(label_dict.keys()))
    with open(groundtruth_dir + video.split("crop")[0] + 'label.txt', 'w') as f:
        for x in range(label_nums):
            f.write(label_dict[str(x + 1)] + '\n')




    #if (not os.path.exists(groundtruth_dir + groundtruth_file)):
    #with open("groundtruth_dir + groundtruth_file)", 'w') as f:









#data = loadmat("/disk2/lzq/data/MERL/Labels_MERL_Shopping_Dataset/1_1_label.mat")
# #data = loadmat("/Users/user/Desktop/Labels_MERL_Shopping_Dataset/1_1_label.mat")
# list = np.array((data['tlabs']))  # shape:(5, 1)
# a = list[0][0]  #  (22, 2)
# label_nums =4200
#
# map_dict = {'0':'background', '1':'reach_to_shelf',
# '2': 'retract_from_shelf',
# '3': 'hand_in_shelf',
# '4':'inspect_product',
# '5': 'inspect_shelf'}
# label_dict = {}
# for i, type in enumerate(list):
#     for action in type[0]:
#         for index in range(action[0], action[1]+1):
#             label_dict[str(index)] = map_dict[str(i+1)]
#     for row in range(label_nums):
#         if str(row+1) not in label_dict.keys():
#             label_dict[str(row+1)] = map_dict['0']
#
# print(len(label_dict.keys()))
# with open("/Users/user/Desktop/test.txt", 'w') as f:
#     for x in range(label_nums):
#         f.write(label_dict[str(x+1)] + '\n')




