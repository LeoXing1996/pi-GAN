"""Datasets"""

import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np
from mmengine import FileClient
import io
from PIL import Image


def load_pil_with_client(path, client):
    img_bytes = client.get(path)
    img_bytes = client.get(path)
    img_bytes = memoryview(img_bytes)
    with io.BytesIO(img_bytes) as buff:
        pil_image = Image.open(buff)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    return pil_image


def load_file_list(path,
                   client: FileClient = None,
                   file_list_path: str = None):
    if file_list_path is not None:
        # TODO:
        file_list = client.get_text(file_list_path)
    else:
        file_list = client.list_dir_or_file(path,
                                            list_dir=False,
                                            recursive=True,
                                            suffix=('png', 'jpg'))
    file_list = [osp.join(path, f) for f in file_list]
    return file_list


class CelebA(Dataset):
    """CelelebA Dataset"""

    def __init__(self,
                 dataset_path,
                 img_size,
                 file_client_args=None,
                 **kwargs):
        super().__init__()

        if file_client_args:
            self.client = FileClient(**file_client_args)
            self.data = load_file_list(dataset_path, self.client)
        else:
            self.client = None
            self.data = glob.glob(dataset_path)

        assert len(self.data) > 0, (
            "Can't find data; make sure you specify the path to your dataset")
        self.transform = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((img_size, img_size), interpolation=0)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.client:
            X = load_pil_with_client(self.data[index], self.client)
        else:
            X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class Cats(Dataset):
    """Cats Dataset"""

    def __init__(self, dataset_path, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, (
            "Can't find data; make sure you specify the path to your dataset")
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.client:
            X = load_pil_with_client(self.data[index], self.client)
        else:
            X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


class Carla(Dataset):
    """Carla Dataset"""

    def __init__(self, dataset_path, img_size, file_client_args=None):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, (
            "Can't find data; make sure you specify the path to your dataset")
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=0),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        if file_client_args is not None:
            self.client = FileClient(**file_client_args)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.client:
            X = load_pil_with_client(self.data[index], self.client)
        else:
            X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X, 0


def get_dataset(name, subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             pin_memory=False,
                                             num_workers=8)
    return dataloader, 3


def get_dataset_distributed(name, world_size, rank, batch_size, **kwargs):
    dataset = globals()[name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    return dataloader, 3
