import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data.dataloader import DataLoader
from typing import Tuple
from torchvision import datasets, transforms


class AlbumDataset(datasets.CIFAR10):
    """
    Wrapper class to use albumentations library with PyTorch Dataset
    """
    def __init__(self, root: str = "./data", train: bool = True, download: bool = True, transform: list = None):
        """
        Constructor
        :param root: Directory at which data is stored
        :param train: Param to distinguish if data is training or test
        :param download: Param to download the dataset from source
        :param transform: List of transformation to be performed on the dataset
        """
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int) -> Tuple:
        """
        Method to return image and its label
        :param index: Index of image and label in the dataset
        """
        image, label = self.data[index], self.targets[index]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, label
    
# Train Phase transformations
train_set_transforms = {
    'randomcrop': A.RandomCrop(height=32, width=32, p=0.2),
    'horizontalflip': A.HorizontalFlip(),
    'cutout': A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=1, min_width=1, fill_value=[0.49139968*255, 0.48215827*255 ,0.44653124*255], mask_fill_value=None),
    'normalize': A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    'standardize': ToTensorV2(),
}

# Test Phase transformations
test_set_transforms = {
    'normalize': A.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),
    'standardize': ToTensorV2()
}
