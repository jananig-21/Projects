from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=4, image_size=32):
    normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kwargs))
