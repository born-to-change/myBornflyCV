import numpy as np
import os
data_dict = np.load("/disk2/lzq/data/MERL/features/6_3.npy")  #(2048, 3921)
print(data_dict.shape)


# for x in os.listdir('/disk2/lzq/data/MERL/groundTruth/'):
#     print(x)
#     with open('/disk2/lzq/data/MERL/groundTruth/'+x, 'a') as f:
#         f.write('background')

