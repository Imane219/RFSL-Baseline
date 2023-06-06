import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from .datasets import register
import pdb


@register('cifar-fs')
class CifarFS(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_tag = split
        split_file = 'CIFAR_FS_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
            # {'catnametolabel':{}, 'labels':[], 'data': []}

        data = pack['data'] #(sample_num,32,32,3)
        label = pack['labels'] #(list[sample_num]), sample_num: 64 for train(random 64 category numbers)

        image_size = 32
        data = [Image.fromarray(x) for x in data] # from array to PIL
        self.data = data
        self.n_classes = len(set(label))
        make_label = []
        # relabel the label numbers with 0-(self.n_classes-1)
        for i in range(self.n_classes):
            make_label.extend([i]*600)
        self.label = make_label

        norm_params = {'mean': [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]],
                       'std': [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size), # scale short edge to image_size
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
                transforms.Resize(image_size),
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
        # dataloader will give the index i, and process PIL image to tensor which values in [0,1]
        return self.transform(self.data[i]), self.label[i], i
