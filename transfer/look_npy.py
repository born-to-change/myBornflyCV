import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

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


#data_dict = np.load("/disk2/lzq/features/breakfest_my2/visual/P03_coffee.npy")  #(921, 2048)

# /disk2/lzq/features/breakfest_my/visual
#  /disk2/lzq/data/breakfast/features/P03_cam01_P03_coffee.npy

# keys = sorted(data_dict.keys())
# for key in keys:
#     weights = data_dict[key][0]
#     biases = data_dict[key][1]
#     print('\n')
#     print(key)
#     print('weights shape', weights.shape)
#     print('biases shape', biases.shape)

# scaler = preprocessing.StandardScaler().fit(data_dict.transpose())
# x = scaler.transform(data_dict.transpose())
#
# norm = preprocessing.MinMaxScaler()
#
# X = norm.fit_transform(data_dict)
# print(X.transpose())


# (2048, 921)   917

import os
import sys
from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc
from tqdm import tqdm

visual_dir = "/disk2/lzq/Videos_MERL_Shopping_Dataset/1_1_crop.mp4"
def is_video(x):
        return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')


vis_existing = [x.split('.')[0] for x in os.listdir(visual_dir)]
# mot_existing = [os.path.splitext(x)[0] for x in os.listdir(motion_dir)]
# flo_existing = [os.path.splitext(x)[0] for x in os.listdir(opflow_dir)]

video_filenames = "1_1_crop"

# Go through each video and extract features

from keras.applications.imagenet_utils import preprocess_input

for video_filename in tqdm(video_filenames):

    # Open video clip for reading
    try:
        clip = VideoFileClip(os.path.join("/disk2/lzq/Videos_MERL_Shopping_Dataset", video_filename))
    except Exception as e:
        sys.stderr.write("Unable to read '%s'. Skipping...\n" % video_filename)
        sys.stderr.write("Exception: {}\n".format(e))
        continue
    shape = (224, 224)
    # Sample frames at 1fps
    fps = int(np.round(clip.fps))
    frames = [scipy.misc.imresize(crop_center(x).astype(np.float32), shape)
              for idx, x in enumerate(clip.iter_frames())]  # if idx % fps == fps // 2

    n_frames = len(frames)

    frames_arr = np.empty((n_frames,) + shape + (3,), dtype=np.float32)
    for idx, frame in enumerate(frames):
        frames_arr[idx, :, :, :] = frame

    frames_arr = preprocess_input(frames_arr)

    # features = model.predict(frames_arr, batch_size=128)
    #
    # name, _ = os.path.splitext(video_filename)
    # feat_filepath = os.path.join(visual_dir, name + '.npy')
    #
    # with open(feat_filepath, 'wb') as f:
    #     np.save(f, features)

plt.figure()
plt.imshow()