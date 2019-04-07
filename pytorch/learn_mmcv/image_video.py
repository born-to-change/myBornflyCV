import mmcv
import numpy as np

img1 = mmcv.imread('1.jpg')

img2 = mmcv.imread('2.jpg', flag='grayscale')

# img = mmcv.imresize(img2, (1000,600), return_scale=False)
#
# img = mmcv.imrotate(img, 30, auto_bound=True)

#img = mmcv.imflip(img2, direction='vertical')  # flip 翻转

# bboxes1 = np.array([10, 10, 100, 120])
# patch = mmcv.imcrop(img1, bboxes1)
#
# # crop two regions (10, 10, 100, 120) and (0, 0, 50, 50)
# bboxes2 = np.array([[10, 10, 100, 120], [0, 0, 50, 50]])
# patches1 = mmcv.imcrop(img1, bboxes2)
#
# # crop two regions, and rescale the patches by 1.2x
# patches2 = mmcv.imcrop(img1, bboxes2)
#
# mmcv.imshow(patches2[1])

# img = mmcv.impad(img2, shape=[600, 1000], pad_val=200)
#
# mmcv.imshow(img)

#video = mmcv.VideoReader('/Users/user/Desktop/datatset/P03_coffee.avi')

# obtain basic informationConcise
# print(len(video))
# print(video.width, video.height, video.resolution, video.fps)
#
#
# # read some frames
# img = video[100]
# mmcv.imshow(img)

#video.cvt2frames('./frames', start=500)

flow = mmcv.flowread('test.flo')
mmcv.flowshow(flow)
