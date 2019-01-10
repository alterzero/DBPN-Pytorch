from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, hr, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolder(hr_dir,patch_size, upscale_factor, data_augmentation,
                             transform=transform())

def get_eval_set(lr_dir, upscale_factor):
    return DatasetFromFolderEval(lr_dir, upscale_factor,
                             transform=transform())

