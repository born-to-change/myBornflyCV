import mmcv
import os
import sys
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from tqdm import tqdm

def is_video(x):
    return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')

'''
Convert a video to frame images, can handle videos_dir(if they have muti video_dir,only can invoke separately) 
or video file and transform frames arganized by filename 

        Args:
           input_path: input video file dir
           output_path: output frames dir
           fps=None: if you want Use frame skipping，assign your own fps(<= video_fps)
           isCrop=False: crop the image 


'''
def video2imgage(input_path, output_path, fps=None, isCrop=False):

    if not os.path.isdir(input_path):
        if is_video(input_path):
            video = mmcv.VideoReader(input_path)
            video.cvt2frames(output_path)
        else:
            sys.stderr.write("Input directory '%s' does not exist!\n" % input_path)
            sys.exit(1)

    #visual_dir = os.path.join(output_dir, 'features')  # RGB features
    # motion_dir = os.path.join(output_dir, 'motion') # Spatiotemporal features
    # opflow_dir = os.path.join(output_dir, 'opflow') # Optical flow features

    # for directory in [visual_dir]:  # , motion_dir, opflow_dir]:
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #
    # vis_existing = [x.split('.')[0] for x in os.listdir(visual_dir)]
    # mot_existing = [os.path.splitext(x)[0] for x in os.listdir(motion_dir)]
    # flo_existing = [os.path.splitext(x)[0] for x in os.listdir(opflow_dir)]

    vis_existing = [x.split('.')[0] for x in os.listdir(input_path)]
    video_filenames = [x for x in sorted(os.listdir(input_path))
                       if is_video(x) and os.path.splitext(x)[0] in vis_existing]
    if fps is not None:
        for video_file in tqdm(video_filenames):
            #  Tqdm a fast Python progress bar tqdm(iterator)
            output_file_path = output_path + '/' + video_file.split('.')[0]
            if os.path.exists(output_file_path):
                pass
            else:
                os.makedirs(output_file_path)
            try:
                clip = VideoFileClip(os.path.join(input_path, video_file))
            except Exception as e:
                sys.stderr.write("Unable to read '%s'. Skipping...\n" % video_file)
                sys.stderr.write("Exception: {}\n".format(e))
                continue

            video_fps = int(np.round(clip.fps))
            if video_fps < fps:
                sys.stderr.write("Unable transfor video to img when your set fps is larger than video ")
            else:
                for idx, x in enumerate(clip.iter_frames()):
                    if (idx % video_fps) % (video_fps // fps) == 0:
                        cv2.imwrite(output_file_path + '/' + str(idx + 1) + '.jpg', x)

    else:
        for video_file in video_filenames:
            output_file_path = output_path + '/' + video_file.split('.')[0]
            if os.path.exists(output_file_path):
                pass
            else:
                os.makedirs(output_file_path)

            video = mmcv.VideoReader(os.path.join(input_path, video_file))
            video.cvt2frames(output_file_path)




    # Go through each video and extract features

video2imgage('/Users/user/Desktop/dataSet/video', '/Users/user/Desktop/dataSet/frame')