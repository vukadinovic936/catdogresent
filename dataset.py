import torch
from torchvision import datasets, transforms, utils
import os
from tqdm import tqdm
import matplotlib as plt
from pathlib import Path
import random
import numpy as np
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def get_data_loaders(data_dir, batch_size = 32, shuffle_val=False):

    # Data augmentation and normalization for training
    # Just normalization for validation

    data_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    image_datasets = ImageFolderWithPaths(data_dir, data_transforms)
    dataset_len = len(image_datasets) 
    train_size = int(dataset_len * 80/100)
    val_size = dataset_len-train_size

    train, val = torch.utils.data.random_split(image_datasets, ((train_size, val_size)))

    # define the dataloaders
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=25, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=shuffle_val, num_workers=25, drop_last =  True)

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {'train': train_size, 'val': val_size}
    class_names = {"cat","dog"}

    print("Class names are")
    print(class_names)

    print(dataset_sizes["train"])
    print(dataset_sizes["val"])

    return dataloaders, dataset_sizes

if __name__ == '__main__':
    dataloaders = get_data_loaders("data/dataset/")
    for i in dataloaders[0]['train']:
        print(i)
        break