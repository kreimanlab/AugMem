import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob
import numpy as np
import random
import torch
from torchvision.datasets.folder import pil_loader

class BatchData(Dataset):
    def __init__(self, images, labels, input_transform=None):
        self.images = images
        self.labels = labels
        self.input_transform = input_transform
        # if self.dataset == 'ilab':
        #     self.dataroot = '/media/data/Datasets/ilab2M/iLab-2M-Light/train_img_distributed' 
        # if self.dataset == 'toybox':
        #     self.dataroot = '/media/data/morgan_data/toybox/images' 
        # if self.dataset == 'core50':
        #     self.dataroot = '/media/data/Datasets/Core50/core50_128x128' #'/media/mengmi/KLAB15/Mengmi/proj_CL_NTM/data/core50/core50_128x128/'
        #if self.dataset == 'cifar100':
        self.dataroot = '/home/rushikesh/P1_Oct/cifar100/cifar100png'
        #self.dataroot = './core50/core50_128x128/'

    def __getitem__(self, index):
        fpath = self.images[index]        
        image = pil_loader(os.path.join(self.dataroot, fpath))        
        #image = Image.fromarray(np.uint8(image))
        label = self.labels[index]
        if self.input_transform is not None:
            image = self.input_transform(image)
        label = torch.LongTensor([label])
        return image, label

    def __len__(self):
        return len(self.images)
