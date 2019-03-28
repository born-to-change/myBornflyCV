import numpy as np


#data = np.load("/disk2/lzq/features/vgg16Test/visual/test1.npy")   #(131, 4096)
#data = np.load("/disk2/lzq/data/breakfast/features/P03_cam01_P03_cereals.npy")    #(2048, 832)
data = np.load("/disk2/lzq/features/resnet50Test/visual/test1.npy")  #(131,2048)
print(data)
print(data.shape)