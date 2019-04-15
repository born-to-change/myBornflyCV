from scipy.io import loadmat
import cv2
import os
import numpy as np
import mmcv

# 0 background
# 1 reach_to_shelf
# 2 retract_from_shelf
# 3 hand_in_shelf
# 4 inspect_product
# 5 inspect_shelf

# video_dir = "/disk2/lzq/Videos_MERL_Shopping_Dataset"
# label_dir = "/disk2/lzq/data/MERL/Labels_MERL_Shopping_Dataset/"
# groundtruth_dir = "/disk2/lzq/data/MERL/groundTruth/"
#
# map_dict = {'0': 'background', '1': 'reach_to_shelf',
#             '2': 'retract_from_shelf',
#             '3': 'hand_in_shelf',
#             '4': 'inspect_product',
#             '5': 'inspect_shelf'}
def gen_all_frames_label(video_dir, label_dir, output_dir, map_dict):
    for video in os.listdir(video_dir):
        video_path = video_dir + '/' + video

        #data_dict = np.load("/disk2/lzq/data/MERL/features/" + video.split("_crop")[0] + '.npy')  # (2048, 3921)

        vid = mmcv.VideoReader(video_path)

        label_nums = len(vid)

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
        with open(output_dir + video.split("_crop")[0] + '.txt', 'w') as f:
            for x in range(label_nums):
                f.write(label_dict[str(x + 1)] + '\n')
            f.close()

    return output_dir



