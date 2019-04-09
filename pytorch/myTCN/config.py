import warnings

class DefaultConfig(object):
    model = 'resnet50'

    train_data_root = ''
    test_data_root = ''
    load_model_path = ''

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20

    epoch = 50
    lr = 5e-5
    lr_decay = 0.95   # lr = lr*lr_decay
    weight_decay = 1e-4


    def parse(self, kwargs):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribute %s' %k)
            setattr(self,k,v)


        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k,getattr(self,k))

# DefaultConfig.parse = parse
opt = DefaultConfig()
