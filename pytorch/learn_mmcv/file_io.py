import mmcv
#
# data = mmcv.load('../scripts/batch_rename.sh')
#
# with open('test.json', file_format = 'json'):
#     mmcv.dump(data, 'out.pkl')

import numpy as np


x = np.loadtxt("/disk2/lzq/charades/Charades_v1_features_rgb/0A8CF/0A8CF-000001.txt")
print(x.shape)

c = mmcv.VideoReader('/disk2/lzq/charades/Charades_v1_480/00HFP.mp4')  #   fps=17
print(c.fps)