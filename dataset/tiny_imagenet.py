import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import  ImageFolder, DatasetFolder

import os
import os.path
import logging
from tqdm import tqdm

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

        self.datas = []
        self.targets = []

        for index in tqdm(range(self.samples.shape[0])):
            path = self.samples[index][0]
            target = self.samples[index][1]
            target = int(target)
            sample = self.loader(path)
            img = np.asarray(sample)
            self.datas.append(img)
            self.targets.append(target)

    def __getitem__(self, index):

        sample = self.datas[index]        
        target = self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)