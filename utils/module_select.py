from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, StepLR
from module.lr_scheduler import CosineAnnealingWarmUpRestarts, YoloLR, Poly

from models.backbone.vgg import vgg16, vgg16_bn
from models.backbone.darknet import darknet19, darknet53
from dataset.classfication.tiny_imagenet import TinyImageNet
from dataset.classfication.tiny_imagenet_v2 import TinyImageNetV2


def get_model(model_name):
    model_dict = {
        'vgg16': vgg16,
        'vgg16_bn': vgg16_bn,
        'darknet19': darknet19,
        'darknet53': darknet53
    }
    return model_dict.get(model_name)


def get_data_module(dataset_name):
    dataset_dict = {
        'tiny-imagenet': TinyImageNet,
        'tiny-imagenet-v2': TinyImageNetV2
    }
    return dataset_dict.get(dataset_name)


def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {
        'sgd': optim.SGD, 
        'adam': optim.Adam,
        'radam': optim.RAdam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop
    }
    optimizer = optim_dict.get(optimizer_name)
    if optimizer:
        return optimizer(params, **kwargs)


def get_scheduler(scheduler_name, optim, **kwargs):
    scheduler_dict = {
        'multi_step': MultiStepLR, 
        'step': StepLR, 
        'cosine_annealing_warm_restarts': CosineAnnealingWarmRestarts,
        'cosine_annealing_warm_up_restarts': CosineAnnealingWarmUpRestarts,
        'yolo_lr': YoloLR,
        'poly': Poly
    }
    optimizer = scheduler_dict.get(scheduler_name)
    if optimizer:
        return optimizer(optim, **kwargs)
