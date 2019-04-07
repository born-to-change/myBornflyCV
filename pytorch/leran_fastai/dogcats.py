from pathlib import Path
from fastai.imports import *
from fastai.basic_train import Learner
from fastai.utils import *
import mmcv


PATH = Path("/Users/user/Desktop/datatset/dogscats/")

bs = 64
sz = 32

print(os.listdir(PATH))

files = os.listdir(f'{PATH}/valid/cats')[:5]

print(files)

#  1. mmcv

# img = mmcv.imread(f'{PATH}/fvalid/cats/{files[0]}')
# mmcv.imshow(img)

#  2. matplotlib.pyplot + PIL

# import matplotlib.pyplot as plt
# from PIL import Image
# img = Image.open(f'{PATH}/valid/cats/{files[0]}')  # f前缀用来格式化字符串,支持在大括号内,运行Python表达式
# gray = img.convert('L')
# r,g,b = img.split()
# img_merged = Image.merge('RGB', (r, g, b))
#
#
# plt.figure(figsize=(10,5)) #设置窗口大小
# plt.suptitle('Multi_Image') # 图片名称
# plt.subplot(2,3,1), plt.title('image')
# plt.imshow(img), plt.axis('off')
# plt.subplot(2,3,2), plt.title('gray')
# plt.imshow(gray,cmap='gray'), plt.axis('off') #这里显示灰度图要加cmap
# plt.subplot(2,3,3), plt.title('img_merged')
# plt.imshow(img_merged), plt.axis('off')
# plt.subplot(2,3,4), plt.title('r')
# plt.imshow(r,cmap='gray'), plt.axis('off')
# plt.subplot(2,3,5), plt.title('g')
# plt.imshow(g,cmap='gray'), plt.axis('off')
# plt.subplot(2,3,6), plt.title('b')
# plt.imshow(b,cmap='gray'), plt.axis('off')
#
# plt.show()


