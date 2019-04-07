from fastai.vision import *
from fastai.metrics import error_rate

bs = 8

path = untar_data(URLs.PETS)

print(path.ls())

path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)  #  PosixPath('/Users/user/.fastai/data/oxford-iiit-pet/images/Egyptian_Mau_167.jpg')
print(fnames[:5])

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'   # 正则表达式，提取图片名

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs).normalize(imagenet_stats)

# data.show_batch(rows=3, figsize=(7, 6))
# plt.show()

print(data.classes)

print(data.c)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)
print(learn.model)
learn.fit_one_cycle(4)

learn.save('stage-1')

interp = ClassificationInterpretation.from_learner(learn)

losses, idxs = interp.top_losses()

interp.plot_top_losses(9, figsize=(15, 11))

doc(interp.plot_top_losses)

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
