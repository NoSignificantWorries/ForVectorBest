import os
import time
import datetime
import json

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset as dset


def train_model(model, criterion, optimizer, num_epochs, save_period=-1):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    save_directory = f"work/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.mkdir(save_directory)
    
    result = {
        "train": {
            "accuracy": [],
            "loss": []
        },
        "valid": {
            "accuracy": [],
            "loss": []
        }
    }

    for epoch in range(num_epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            with tqdm(dataloaders[phase], desc=f"{phase} Epoch {epoch + 1}/{num_epochs}") as pbar:
                for inputs, labels in pbar:
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
                    pbar.set_postfix({"loss": epoch_loss, "acc": epoch_acc.item()})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            result[phase]["accuracy"].append(epoch_acc.item())
            result[phase]["loss"].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        if save_period > 0:
            if (epoch + 1) % save_period == 0:
                torch.save(model.state_dict(), f"{save_directory}/resnet50_{epoch + 1}.pth")
            

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    torch.save(best_model_wts, f"{save_directory}/resnet50_best.pth")
    torch.save(model.state_dict(), f"{save_directory}/resnet50_last.pth")

    with open(f"{save_directory}/results.json", "w") as json_file:
        json.dump(result, json_file, indent=4)
    return result
        

if __name__ == "__main__":
    # Loading data {{{
    image_datasets = {
        "train": dset.StickersDataset("stickers/train", transform=dset.DatasetParams.data_transforms["train"]),
        "valid": dset.StickersDataset("stickers/valid", transform=dset.DatasetParams.data_transforms["valid"])
    }
    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=4, shuffle=True, num_workers=2),
        "valid": DataLoader(image_datasets["valid"], batch_size=4, shuffle=False, num_workers=2)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}
    # }}}

    # Model preparing {{{
    model_ft = models.resnet50(pretrained=True)

    # Freezing pretrained layers
    for param in model_ft.parameters():
        param.requires_grad = False

    # Replacing final layer with our classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dset.DatasetParams.classes))

    # Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)
    # }}}
    
    # Training model {{{
    train_results = train_model(model_ft, criterion, optimizer_ft, num_epochs=200, save_period=10)
    # }}}
