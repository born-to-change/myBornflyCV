import numpy as np
import os

input_path = '/disk2/lzq/charades/Charades_v1_features_rgb'
output_path = '/disk2/lzq/data/charades/features'

for video in os.listdir(input_path):
    features = []
    for frame in os.listdir(input_path+'/'+video):
        row = np.loadtxt(input_path+'/'+video+'/'+frame)
        features.append(row)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    feat_filepath = os.path.join(output_path, video + '.npy')

    with open(feat_filepath, 'wb') as f:
        np.save(f, features)