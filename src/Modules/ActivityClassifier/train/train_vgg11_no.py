import torch
import torch.nn as nn
import numpy as np
import os
import torchvision
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm


# ================================================
# UTILS
# ================================================
# 정확도 측정
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Validation 평가
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = []
    for batch in val_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        outputs.append({"val_loss": loss.detach(), "val_acc": acc})

    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine Losses
    batch_accs = [x["val_acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine Accuracies
    return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}


# ================================================
# CONFIG
# ================================================
PRE_WEIGHT = './weight/vgg11/vgg11_7.pth'
WEIGHT_PATH = './weight/vgg11/new/'
WEIGHT_NAME = 'vgg11'
DATA_NAME = 'activity'
DATA_PATH = f'./dataset/{DATA_NAME}/'
INPUT_SIZE = 3
NUM_CLASSES = 10


# ================================================
# Data 개수 확인
# ================================================
train_length = 0
for clss in os.listdir(DATA_PATH):
    print("%s size: %d" % (clss, len(os.listdir(os.path.join(DATA_PATH, clss)))))
    train_length += len(os.listdir(os.path.join(DATA_PATH, clss)))
print("Train size: %d" % train_length)


# ================================================
# HyperParameter 설정
# ================================================
EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 32


# ================================================
# Data Augmentation
# ================================================
train_transforms = T.Compose([T.Resize((64, 64)),
                              T.RandomAdjustSharpness(2),
                              T.RandomRotation((-15, 15)),
                              T.ColorJitter(brightness=.5, hue=.3),
                              T.ToTensor()])


# ================================================
# Data Loader 생성
# ================================================
train_ds = ImageFolder(DATA_PATH, train_transforms)
classes = train_ds.classes
print(classes)

# Validation Set 수량 설정
val_pct = .1
val_size = int(val_pct * len(train_ds))
train_ds, valid_ds = random_split(train_ds, [len(train_ds)-val_size, val_size])

# Data Loader
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds, BATCH_SIZE, num_workers=2, pin_memory=True)


# ================================================
# CUDA Device 설정
# ================================================
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
torch.cuda.empty_cache()


# ================================================
# Model 설정
# ================================================
# model = models.vgg11(pretrained=True)
model = models.vgg11()
# model.load_state_dict(torch.load(PRE_WEIGHT, map_location=device))
model = model.to(device)

# Multi GPU 설정
model = nn.DataParallel(model, device_ids=[0, 1, 2]).cuda()


# ================================================
# Optimizer / Criterion
# ================================================
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().cuda()


# ================================================
# Training
# ================================================
print('\n================================================')
print('Train Start')
print('================================================')

torch.cuda.empty_cache()
history = []
for epoch in range(EPOCHS):
    # Training Phase
    model.train()
    train_losses = []
    for batch in tqdm(train_dl):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)

        loss = criterion(out, labels)
        train_losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation Phase
    result = evaluate(model, valid_dl)
    result["train_loss"] = torch.stack(train_losses).mean().item()
    print("Epoch [{}/{}], train_loss : {:.4f}, val_loss : {:.4f}, val_acc : {:.4f}".format(epoch, EPOCHS,
                                                                                           result["train_loss"],
                                                                                           result["val_loss"],
                                                                                           result["val_acc"]))
    torch.save(model.module.state_dict(), WEIGHT_PATH + f'{WEIGHT_NAME}_{epoch}.pth')
    # torch.save(model.state_dict(), WEIGHT_PATH + f'{WEIGHT_NAME}_{epoch}.pth')
    history.append(result)

