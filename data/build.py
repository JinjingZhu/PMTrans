# --------------------------------------------------------
# Reference from https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------

import os
from random import shuffle

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision import transforms

from .samplers import SubsetRandomSampler

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


class ObjectImage_mul(torch.utils.data.Dataset):
    """
    :return img, label, index for pseudo labels
    """

    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__ == 'list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def build_loader(config):
    dsets = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dset_loaders = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dsets['source_train'], dsets['target_train'] = build_dataset(is_train=True, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")

    dsets['source_val'], dsets['target_val'] = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train_source = torch.utils.data.DistributedSampler(
        dsets['source_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_train_target = torch.utils.data.DistributedSampler(
        dsets['target_train'], num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices_t = np.arange(dist.get_rank(), len(dsets['target_val']), dist.get_world_size())
    sampler_val_t = SubsetRandomSampler(indices_t)

    indices_s = np.arange(dist.get_rank(), len(dsets['source_val']), dist.get_world_size())
    sampler_val_s = SubsetRandomSampler(indices_s)

    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], sampler=sampler_train_source,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], sampler=sampler_train_target,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'], sampler=sampler_val_s,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    dset_loaders['target_val'] = torch.utils.data.DataLoader(
        dsets['target_val'], sampler=sampler_val_t,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    return dsets, dset_loaders

def build_loader_parallel(config):
    dsets = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dset_loaders = {
        'target_train': {},
        'source_train': {},
        'target_val': {},
        'source_val': {},
    }
    dsets['source_train'], dsets['target_train'] = build_dataset(is_train=True, config=config)
    print(f"Successfully build train dataset")

    dsets['source_val'], dsets['target_val'] = build_dataset(is_train=False, config=config)
    print(f"Successfully build val dataset")


    dset_loaders['source_train'] = torch.utils.data.DataLoader(
        dsets['source_train'], shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['target_train'] = torch.utils.data.DataLoader(
        dsets['target_train'], shuffle=True,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    dset_loaders['source_val'] = torch.utils.data.DataLoader(
        dsets['source_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    dset_loaders['target_val'] = torch.utils.data.DataLoader(
        dsets['target_val'],
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True
    )

    return dsets, dset_loaders

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if is_train:
        source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
        source_dataset = ObjectImage('', source_root, transform)
        target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
        target_dataset = ObjectImage_mul('', target_root, transform)
        return source_dataset, target_dataset
    else:
        source_root = os.path.join(config.DATA.DATA_PATH, config.DATA.SOURCE + '.txt')
        source_dataset = ObjectImage('', source_root, transform)
        target_root = os.path.join(config.DATA.DATA_PATH, config.DATA.TARGET + '.txt')
        target_dataset = ObjectImage('', target_root, transform)
        return source_dataset, target_dataset


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    mean = config.DATA.MEAN
    std = config.DATA.STD
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.RandomCrop(config.DATA.IMG_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((config.DATA.IMG_SIZE + 32, config.DATA.IMG_SIZE + 32)),
            transforms.CenterCrop((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform


class ResizeImage:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))
