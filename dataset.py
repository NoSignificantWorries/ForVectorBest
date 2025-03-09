import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class StickersDataset(Dataset):
    def __init__(self, root_dir, classes=["Good", "Bad"], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes

        self.images = []
        self.labels = []

        for i, image_class in enumerate(classes):
            class_path = os.path.join(root_dir, image_class)
            for image_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, image_name))
                self.labels.append(i)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_obj = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform is not None:
            image_obj = self.transform(image_obj)

        return image_obj, label


def calculate_mean_std(dataset, batch_size=6, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images in dataloader:
        batch_size = images[0].size(0)
        images = images[0].view(batch_size, images[0].size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_size

    mean /= total_samples
    std /= total_samples

    return mean, std


def get_mean_std():
    valid = StickersDataset("stickers/valid")
    test = StickersDataset("stickers/test")
    full = StickersDataset("stickers/train", transform=transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor()]))
    full.images += test.images + valid.images
    full.labels += test.labels + valid.labels
    print(calculate_mean_std(full))


class DatasetParams:
    classes = ["Good", "Bad"]
    mean = [0.6755, 0.6786, 0.6398]
    std = [0.1850, 0.1585, 0.1613]
    
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
