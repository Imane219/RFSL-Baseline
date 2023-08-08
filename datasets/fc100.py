# https://github.com/yaoyao-liu/meta-transfer-learning/blob/main/pytorch/dataloader/dataset_loader.py
import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from .datasets import register
import pdb


@register('fc100')
class FC100(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        root_path = os.path.join(root_path, split)
        label_list = os.listdir(root_path)

        data = []
        label = []
        folders = [os.path.join(root_path, label) for label in label_list if os.path.isdir(os.path.join(root_path, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(os.path.join(this_folder, image_path))
                label.append(idx)

        data = [Image.open(x).convert('RGB') for x in data]#从array变成PIL

        self.data = data
        self.label = label
        self.n_classes = len(set(label))

        image_size = 84

        #CIFAR10
        # norm_params = {'mean': [x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                 'std': [x / 255.0 for x in [63.0, 62.1, 66.7]]}

        norm_params = {'mean': [0.507, 0.487, 0.441], 'std': [0.267, 0.256, 0.276]}
        normalize = transforms.Normalize(**norm_params)
        # self.default_transform = transforms.Compose([
        #     transforms.Resize(image_size),
        #     transforms.ToTensor(),
        #     normalize,  
        # ])
        self.default_transform = transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),#scale short edge to image_size
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

        def normalize_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return (x - mean) / std
        self.normalize_raw = normalize_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i], i
