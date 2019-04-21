import json
import os
import numpy as np

feature_path = '/disk2/lzq/data/charades/features'


with open('/home/zhuoqun/Prj/myBornflyCV/pytorch/data/Charades/charades.json') as f:
    data = json.load(f)
    i = 0
    for key in data.keys():
        i +=1
        print(key)
        print(data[key])
        if data[key]['subset'] == "training":
            features = np.load(feature_path + '/' + key + '.npy')
            num = features[0]
            all_num = 24*data[key]
            dict = {}
            for action in data[key]['actions']:





    print(i)
