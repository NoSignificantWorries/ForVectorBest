import os
import time

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


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


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model
        

if __name__ == "__main__":
    # Calculatind mean and std for dataset {{{
    '''
    valid = StickersDataset("stickers/valid")
    test = StickersDataset("stickers/test")
    full = StickersDataset("stickers/train", transform=transforms.Compose([transforms.Resize([256, 256]), transforms.ToTensor()]))
    full.images += test.images + valid.images
    full.labels += test.labels + valid.labels
    print(calculate_mean_std(full))
    '''
    # Actual metrics are:
    mean = [0.6755, 0.6786, 0.6398]
    std = [0.1850, 0.1585, 0.1613]
    # }}}
    
    # Data transformers for dataset {{{
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
    # }}}
    
    # Loading data {{{
    image_datasets = {
        "train": StickersDataset("stickers/train", transform=data_transforms["train"]),
        "valid": StickersDataset("stickers/valid", transform=data_transforms["valid"])
    }
    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=4, shuffle=True, num_workers=2),
        "valid": DataLoader(image_datasets["valid"], batch_size=4, shuffle=False, num_workers=2)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}
    classes = ["Good", "Bad"]
    # }}}

    # Model preparing {{{
    model_ft = models.resnet50(pretrained=True)

    '''
    Freezing pretrained layers
    '''
    for param in model_ft.parameters():
        param.requires_grad = False

    '''
    Replacing final layer with our classes
    '''
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)
    # }}}
    
    # Training model {{{
    model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=50)
    torch.save(model_ft.state_dict(), 'resnet50_finetuned.pth')
    # }}}
