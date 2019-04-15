import os
from utils import genAllFrammesLabel
import mmcv

def gen_merl_data(imgpath, video_path, ann_path, split_path):
    map_dict = {'0': 'background', '1': 'reach_to_shelf',
                '2': 'retract_from_shelf',
                '3': 'hand_in_shelf',
                '4': 'inspect_product',
                '5': 'inspect_shelf'}
    output_dir = ann_path + '/annotatiuons'
    if os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_path = genAllFrammesLabel(video_path, ann_path, output_dir, map_dict)
    for split_file in os.path.isdir(split_path):
        if split_file.split('.')[0] == 'train':
            with open(split_path + split_file, 'r') as f:
                data = mmcv.load(f)






