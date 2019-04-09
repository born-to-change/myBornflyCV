from fastai.script import *
from fastai.vision import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callbacks.tracker import *
torch.backend.cudnn.benchmark = True

# def get_data(patch, size, workers=None, use_lighting=False):
#     path = Path('')
#     num_gpus = num_distrib() or 1
#     if workers is None:
#         workers = min(8, num_gpus()//num_gpus)
#         tfms = [flip_lr(p=0.5)]
#     if use_lighting:
#         tfms += [brightness(change=(0.4,0.6)), contrast(scale=(0.7,1.3))]
#     return [ImageList.from_folder(path).split_by_folder(valid='val')
#             .label_from_folder().transform((tfms, []), size=size)
#             .databunch(bs=bs, num_workers=workers)
#             .presize(size, scale=(0,35,1))
#             .normalize(imagenet_stats)]
#
#
#
# def get_learner(lr, size, woof, bs, opt_func, gpu, epochs):
#     data = get_data(size, woof, bs)
#     bs_rat = bs/256
#     lr *= bs_rat
#     b_its = len(data.train_dl)
#
#     ph1 = (TrainingPhase(epochs*0.5*b_its)
#             .schedule_hp('lr', (lr/20,lr), anneal=annealing_cos)
#             .schedule_hp('eps', (1e-4,1e-7), anneal=annealing_cos)
#             )
#     ph2 = (TrainingPhase(epochs*0.5*b_its)
#             .schedule_hp('lr', (lr,lr/1e5), anneal=annealing_cos)
#             .schedule_hp('eps', (1e-7,1e-7), anneal=annealing_cos)
#             )
#
#     learn = (Learner(data, models.xresnet50(),
#              metrics=[accuracy,top_k_accuracy], wd=1e-3, opt_func=opt_func,
#              bn_wd=False, true_wd=True, loss_func = LabelSmoothingCrossEntropy())
#         .mixup(alpha=0.2)
#         .to_fp16(dynamic=True)
#     )
#     if gpu is None:       learn.to_parallel()
#     elif num_distrib()>1: learn.to_distributed(gpu)
#
#     gs = GeneralScheduler(learn, (ph1,ph2))
#     learn.fit(epochs, lr=1, callbacks=gs)
# from fastai.script import *
# from fastai.vision import *
# from fastai.vision.models.wrn import wrn_22
# from fastai.distributed import *
# torch.backends.cudnn.benchmark = True
#
#
# @call_parse
# def main( gpu:Param("GPU to run on", str)=None ):
#     """Distrubuted training of CIFAR-10.
#     Fastest speed is if you run as follows:
#         python -m fastai.launch train_cifar.py"""
#     gpu = setup_distrib(gpu)
#     n_gpus = num_distrib()
#     path = url2path(URLs.CIFAR)
#     ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
#     workers = min(16, num_cpus()//n_gpus)
#     data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512//n_gpus,
#                                       num_workers=workers).normalize(cifar_stats)
#     learn = Learner(data, wrn_22(), metrics=accuracy)
#     if gpu is None: learn.model = nn.DataParallel(learn.model)
#     else: learn.to_distributed(gpu)
#     learn.to_fp16()
#     learn.fit_one_cycle(35, 3e-3, wd=0.4)
#

data = ImageList.