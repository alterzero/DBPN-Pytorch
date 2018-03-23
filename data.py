from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from dataset import DatasetFromFolderEval, DatasetFromFolder

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform():
    return Compose([
        #CenterCrop(crop_size),
        #Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform():
    return Compose([
        #CenterCrop(crop_size),
        ToTensor(),
    ])


def get_training_set(data_dir, dataset, hr, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    lr_dir = join(data_dir, dataset)
    #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(hr_dir, lr_dir,patch_size, upscale_factor, dataset, data_augmentation,
                             input_transform=input_transform(),
                             target_transform=target_transform())


def get_test_set(data_dir, dataset, hr, upscale_factor,patch_size):
    hr_dir = join(data_dir, hr)
    lr_dir = join(data_dir, dataset)
    #crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

    return DatasetFromFolder(hr_dir, lr_dir,patch_size, upscale_factor, dataset, data_augmentation=False,
                             input_transform=input_transform(),
                             target_transform=target_transform())

def get_eval_set(lr_dir):
    return DatasetFromFolderEval(lr_dir,
                             input_transform=input_transform(),
                             target_transform=target_transform())

